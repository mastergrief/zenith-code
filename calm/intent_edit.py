"""
Auto-CALM Layer 3 — intent-to-edit: NL diagnosis → template fix → verify.

3-step bug fixing:
1. DIAGNOSE — model reads code + test failures, describes bugs
2. GENERATE — engine applies deterministic templates or LLM full-rewrite
3. VERIFY — run tests, self-heal on remaining failures

Usage:
    from calm.intent_edit import IntentToEdit
    result = IntentToEdit().fix("app.py", "test_app.py", verbose=True)
"""

from __future__ import annotations

import ast as _ast
import json
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional


EDIT_SYSTEM_PROMPT = """\
You are a code repair assistant. You fix bugs in Python code.
Be precise. Output ONLY the replacement code when asked."""


@dataclass
class EditResult:
    original_tests: str = ""
    final_tests: str = ""
    edits_applied: int = 0
    edits_attempted: int = 0
    diagnosis: str = ""
    steps: List[dict] = field(default_factory=list)
    success: bool = False


@dataclass
class EditIntent:
    file: str
    line: int
    action: str
    description: str
    code: str = ""


class IntentToEdit:
    """3-step bug fixer: diagnose → template/LLM fix → verify."""

    def __init__(self, server="http://localhost:8080",
                 max_tokens=16384, thinking_budget=32768):
        self.server = server
        self.max_tokens = max_tokens
        self.thinking_budget = thinking_budget

    def fix(self, file_path: str, test_path: str, verbose=False) -> EditResult:
        import os
        result = EditResult()
        test_dir = os.path.dirname(os.path.abspath(test_path))

        baseline = self._run_tests(test_path, cwd=test_dir)
        result.original_tests = baseline
        if verbose:
            print(f"[step 0] baseline: {baseline}")

        source = open(file_path).read()

        # Step 1: DIAGNOSE.
        diagnosis = self._diagnose(source, file_path, baseline, verbose)
        result.diagnosis = diagnosis
        if verbose:
            print(f"[step 1] diagnosis: {len(diagnosis)} chars")
            print(f"  {diagnosis[:300].replace(chr(10), ' ')}...")

        # Step 2: GENERATE — extract intents, generate code.
        intents = self._extract_intents(diagnosis, file_path)
        result.edits_attempted = len(intents)
        if verbose:
            print(f"[step 2] {len(intents)} edit intents extracted")

        if not intents:
            if verbose:
                print(f"[step 2] no edits extracted — full rewrite")
            fixed = self._generate_full_fix(source, file_path, baseline, verbose, diagnosis)
            if fixed and fixed != source:
                with open(file_path, 'w') as f:
                    f.write(fixed)
                result.edits_applied = 1
                after = self._run_tests(test_path, cwd=test_dir)
                result.final_tests = after
                result.success = "failed" not in after.lower() or \
                    self._count_passed(after) > self._count_passed(baseline)
                if verbose:
                    print(f"[step 3] after fix: {after}")
                return result

        # Generate replacement code per intent.
        for intent in intents:
            intent.code = self._generate_code(intent, source, verbose)
            if verbose:
                print(f"  [{intent.action}] line {intent.line}: {intent.description[:60]}")

        # Apply edits bottom-to-top.
        lines = source.splitlines()
        for intent in sorted([i for i in intents if i.code], key=lambda i: i.line, reverse=True):
            idx = intent.line - 1
            if idx < 0 or idx >= len(lines):
                continue
            if intent.action == "replace":
                lines[idx] = intent.code
            elif intent.action == "insert_before":
                lines.insert(idx, intent.code)
            elif intent.action == "insert_after":
                lines.insert(idx + 1, intent.code)
            elif intent.action == "wrap":
                lines[idx:idx+1] = intent.code.split('\n')
            result.edits_applied += 1

        with open(file_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')

        # Syntax check — revert + full rewrite on failure.
        try:
            _ast.parse('\n'.join(lines))
        except SyntaxError:
            if verbose:
                print(f"[step 3] syntax error — trying full rewrite")
            with open(file_path, 'w') as f:
                f.write(source)
            fixed = self._generate_full_fix(source, file_path, baseline, verbose, diagnosis)
            if fixed:
                with open(file_path, 'w') as f:
                    f.write(fixed)
            else:
                result.final_tests = "syntax error — could not fix"
                return result

        # Run tests.
        after = self._run_tests(test_path, cwd=test_dir)
        result.final_tests = after
        result.success = "failed" not in after.lower() or \
            self._count_passed(after) > self._count_passed(baseline)
        if verbose:
            print(f"[step 3] after fix: {after}")

        # Regression → revert + full rewrite.
        if self._count_passed(after) < self._count_passed(baseline):
            if verbose:
                print(f"[step 3] regression — full rewrite")
            with open(file_path, 'w') as f:
                f.write(source)
            fixed = self._generate_full_fix(source, file_path, baseline, verbose, diagnosis)
            if fixed:
                with open(file_path, 'w') as f:
                    f.write(fixed)
                after = self._run_tests(test_path, cwd=test_dir)
                result.final_tests = after
                result.success = self._count_passed(after) > self._count_passed(baseline)

        # Training data collection.
        if self._count_passed(after) > self._count_passed(baseline):
            from calm.auto_training import AutoTrainingCollector
            tc = AutoTrainingCollector()
            tc.collect_from_edit(file_path, diagnosis, source,
                                open(file_path).read(), baseline, after)

        # Self-healing retry (max 1).
        if self._count_passed(after) < 10 and "failed" in after.lower():
            current = open(file_path).read()
            if verbose:
                print(f"[step 4] self-healing retry")
            fixed2 = self._generate_full_fix(current, file_path, after, verbose)
            if fixed2:
                with open(file_path, 'w') as f:
                    f.write(fixed2)
                after2 = self._run_tests(test_path, cwd=test_dir)
                if self._count_passed(after2) >= self._count_passed(after):
                    result.final_tests = after2
                    result.success = "failed" not in after2.lower()
                    if verbose:
                        print(f"[step 4] after retry: {after2}")
                else:
                    with open(file_path, 'w') as f:
                        f.write(current)

        return result

    # --- Private helpers ---

    def _diagnose(self, source, path, test_output, verbose):
        content, _, _ = self._generate([
            {"role": "system", "content": EDIT_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Here is `{path}`:\n```python\n{source}\n```\n\n"
                f"Test results:\n```\n{test_output}\n```\n\n"
                f"Describe each bug. For each: line number, what's wrong, "
                f"replacement code. Format: 'Line N: replace with `code`'."
            )},
        ])
        return content

    def _extract_intents(self, diagnosis, file_path):
        intents = []
        for m in re.finditer(
            r'[Ll]ine\s+(\d+).*?(?:replace|change|modify|update)\s+.*?'
            r'(?:with|to)\s*[`\'"](.*?)[`\'"]', diagnosis, re.DOTALL):
            intents.append(EditIntent(file=file_path, line=int(m.group(1)),
                                     action="replace", description=m.group(0)[:200]))
        for m in re.finditer(
            r'[Ll]ine\s+(\d+).*?add\s+[`\'"](.*?)[`\'"]\s*(before|after)', diagnosis):
            action = "insert_before" if m.group(3) == "before" else "insert_after"
            intents.append(EditIntent(file=file_path, line=int(m.group(1)),
                                     action=action, description=m.group(0)[:200]))
        for m in re.finditer(
            r'[Ll]ine\s+(\d+).*?(?:wrap|surround|enclose)\s+.*?(?:in|with)\s+(?:a\s+)?(\w+)',
            diagnosis):
            intents.append(EditIntent(file=file_path, line=int(m.group(1)),
                                     action="wrap", description=m.group(0)[:200]))
        return intents

    def _generate_code(self, intent, source, verbose):
        lines = source.splitlines()
        start = max(0, intent.line - 4)
        end = min(len(lines), intent.line + 3)
        context = '\n'.join(
            f"{'>>>' if i == intent.line - 1 else '   '} {i+1}: {lines[i]}"
            for i in range(start, end))
        content, _, _ = self._generate([
            {"role": "system", "content": EDIT_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Edit: {intent.description}\n\nContext:\n{context}\n\n"
                f"Output ONLY replacement code for line {intent.line}. No explanation.")},
        ])
        code = content.strip()
        code = re.sub(r'^```\w*\n?', '', code)
        code = re.sub(r'\n?```$', '', code).strip()
        if '\n' in code:
            code_lines = [l for l in code.splitlines() if l.strip()]
            if code_lines:
                code = code_lines[0]
        return code

    def _apply_template_fixes(self, source, test_output, verbose):
        lines = source.splitlines()
        modified = False

        if "ZeroDivisionError" in test_output:
            for i, line in enumerate(lines):
                stripped = line.strip()
                indent = line[:len(line) - len(line.lstrip())]
                if ('/ b' in stripped or '/ y' in stripped or '/b' in stripped) \
                   and 'return' in stripped and '== 0' not in stripped:
                    m = re.search(r'/\s*(\w+)', stripped)
                    if m:
                        var = m.group(1)
                        lines.insert(i, f'{indent}if {var} == 0:\n{indent}    return "Error: division by zero"')
                        modified = True
                        if verbose: print(f"  [template] line {i+1}: zero-check for {var}")
                        break

        if "ValueError" in test_output and "float" in test_output:
            for i, line in enumerate(lines):
                stripped = line.strip()
                indent = line[:len(line) - len(line.lstrip())]
                if 'return float(' in stripped or 'return int(' in stripped:
                    lines[i] = (f'{indent}try:\n{indent}    {stripped}\n'
                                f'{indent}except (ValueError, TypeError):\n'
                                f'{indent}    return "Error: invalid input"')
                    modified = True
                    if verbose: print(f"  [template] line {i+1}: try/except")
                    break

        if "IndexError" in test_output:
            max_idx, list_var, first_line = 0, None, None
            for i, line in enumerate(lines):
                for m in re.finditer(r'(\w+)\[(\d+)\]', line.strip()):
                    var, idx = m.group(1), int(m.group(2))
                    if idx >= max_idx:
                        max_idx, list_var = idx, var
                    if first_line is None:
                        first_line = i
            if list_var and first_line is not None:
                indent = lines[first_line][:len(lines[first_line]) - len(lines[first_line].lstrip())]
                lines.insert(first_line, f'{indent}if len({list_var}) < {max_idx + 1}:\n'
                             f'{indent}    return "Error: malformed input"')
                modified = True
                if verbose: print(f"  [template] line {first_line+1}: bounds-check for {list_var}[0..{max_idx}]")

        if not modified:
            return None
        result = '\n'.join(lines) + '\n'
        try:
            _ast.parse(result)
            return result
        except SyntaxError:
            return None

    def _generate_full_fix(self, source, path, test_output, verbose, diagnosis=""):
        tmpl = self._apply_template_fixes(source, test_output, verbose)
        if tmpl:
            if verbose: print(f"  [template] applied deterministic fix")
            return tmpl

        diag = f"\nYour diagnosis:\n{diagnosis}\n" if diagnosis else ""
        content, _, _ = self._generate([
            {"role": "system", "content": EDIT_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Here is the COMPLETE `{path}` ({len(source.splitlines())} lines):\n"
                f"```python\n{source}\n```\n\n"
                f"Test failures:\n```\n{test_output[-500:]}\n```\n{diag}\n"
                f"Output the COMPLETE fixed file with ALL functions. "
                f"Only modify buggy parts. ONLY Python code, no markdown.")},
        ])
        code = content.strip()
        code = re.sub(r'^```\w*\n?', '', code)
        code = re.sub(r'\n?```$', '', code).strip()
        try:
            _ast.parse(code)
            return code + '\n'
        except SyntaxError:
            if verbose: print(f"[full-fix] syntax error in generated code")
            return None

    def _run_tests(self, test_path, cwd=None):
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", test_path, "-v", "--tb=short", "-q"],
                capture_output=True, text=True, timeout=30, cwd=cwd)
            return proc.stdout[-500:] + proc.stderr[-200:]
        except subprocess.TimeoutExpired:
            return "timeout"

    def _count_passed(self, test_output):
        m = re.search(r'(\d+) passed', test_output)
        return int(m.group(1)) if m else 0

    def _generate(self, messages):
        payload = {"messages": messages, "temperature": 0.0,
                   "max_tokens": self.max_tokens, "stream": False}
        if self.thinking_budget > 0:
            payload["enable_thinking"] = True
            payload["thinking_budget"] = self.thinking_budget
        req = urllib.request.Request(
            f"{self.server}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
        choice = data["choices"][0]
        return (choice["message"].get("content", ""),
                choice["message"].get("reasoning_content", ""),
                data.get("timings", {}))
