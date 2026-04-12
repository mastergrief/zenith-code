"""
CALM type analysis backend — computed type annotation coverage.

Models suggest wrong types. This backend checks what's annotated,
what's not, and validates existing annotations against AST structure.

Functions: annotation_coverage, unannotated_functions, type_summary,
return_types, parameter_types.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List


def annotation_coverage(source: str) -> dict:
    """Measure type annotation coverage.
    Returns {total_functions, annotated, unannotated, coverage_pct, rating}."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {"error": "syntax error"}

    total = 0
    annotated = 0
    unannotated_list = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_") and node.name != "__init__":
                continue  # Skip private helpers.
            total += 1

            has_return = node.returns is not None
            params = [a for a in node.args.args if a.arg != "self"]
            has_param_annotations = all(a.annotation is not None for a in params) if params else True

            if has_return and has_param_annotations:
                annotated += 1
            else:
                missing = []
                if not has_return:
                    missing.append("return type")
                unannotated_params = [a.arg for a in params if a.annotation is None]
                if unannotated_params:
                    missing.append(f"params: {', '.join(unannotated_params)}")
                unannotated_list.append({
                    "name": node.name,
                    "line": node.lineno,
                    "missing": missing,
                })

    coverage = round(annotated / total * 100, 1) if total > 0 else 100

    return {
        "total_functions": total,
        "annotated": annotated,
        "unannotated": total - annotated,
        "coverage_pct": coverage,
        "rating": (
            "fully typed" if coverage == 100 else
            "well typed" if coverage >= 80 else
            "partially typed" if coverage >= 50 else
            "mostly untyped"
        ),
        "unannotated_functions": unannotated_list[:10],
    }


def annotation_coverage_file(path: str) -> dict:
    """Run annotation_coverage on a file."""
    source = Path(path).read_text(encoding="utf-8", errors="replace")
    result = annotation_coverage(source)
    result["file"] = path
    return result


def return_types(source: str) -> list:
    """Analyze return types per function.
    Returns [{name, line, has_annotation, return_annotation, returns_none}]."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [{"error": "syntax error"}]

    results = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            annotation = None
            if node.returns:
                annotation = ast.dump(node.returns)
                # Clean up common patterns.
                annotation = annotation.replace("Constant(value=None)", "None")
                if "Name(id='" in annotation:
                    import re
                    annotation = re.search(r"Name\(id='(\w+)'\)", annotation)
                    annotation = annotation.group(1) if annotation else "complex"

            # Check if function has explicit return None or no return.
            has_return = any(isinstance(n, ast.Return) and n.value is not None
                           for n in ast.walk(node))

            results.append({
                "name": node.name,
                "line": node.lineno,
                "has_annotation": node.returns is not None,
                "return_annotation": annotation,
                "has_return_value": has_return,
            })

    return results


def parameter_types(source: str) -> list:
    """Analyze parameter type annotations per function.
    Returns [{name, line, params: [{name, annotated, annotation}]}]."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [{"error": "syntax error"}]

    results = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = []
            for arg in node.args.args:
                if arg.arg == "self":
                    continue
                annotation = None
                if arg.annotation:
                    if isinstance(arg.annotation, ast.Name):
                        annotation = arg.annotation.id
                    elif isinstance(arg.annotation, ast.Constant):
                        annotation = str(arg.annotation.value)
                    else:
                        annotation = "complex"

                params.append({
                    "name": arg.arg,
                    "annotated": arg.annotation is not None,
                    "annotation": annotation,
                })

            results.append({
                "function": node.name,
                "line": node.lineno,
                "params": params,
                "all_annotated": all(p["annotated"] for p in params) if params else True,
            })

    return results


TYPE_FUNCTIONS = {
    "annotation_coverage": annotation_coverage,
    "annotation_coverage_file": annotation_coverage_file,
    "return_types": return_types,
    "parameter_types": parameter_types,
}
