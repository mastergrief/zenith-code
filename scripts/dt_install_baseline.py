"""Measure Gemma's natural signature-emission baseline vs ground truth.

Context: DT is trained to predict `def FN(<args>):` skeletons from NL.
v14 hits 0.20 greedy / 0.314 beam honest val. We don't know if Gemma's
natural signature emission is ABOVE or BELOW that. This script measures.

Procedure:
1. Load 50 honest-val (prompt, skeleton) pairs from same split as DT
   training.
2. For each prompt, ask Gemma to emit a function signature for it.
3. Parse Gemma's first-line signature, canonicalize (replace fn name
   with FN), compare arg list to ground truth.
4. Report Gemma's signature-match rate.

Runs via gemma-run daemon. Output at DONE marker.
"""
from __future__ import annotations

import re
import sys
import torch

from calm.hrm.code_dt_data import extract_pairs_from_db, split_pairs
from calm.hrm.rare_class_synth import canonical_arg, parse_arg_names

pairs = extract_pairs_from_db(augment=False, extract_all_defs=True)
_, val = split_pairs(pairs, val_frac=0.1, seed=42)
# Use first 50 for speed
val = val[:50]
print(f"[baseline] measuring on {len(val)} val prompts")

# `m` and `tok` are injected by gemma_daemon; see other scripts in
# scripts/r53_*.py for the pattern
if "m" not in globals() or "tok" not in globals():
    print("ERROR: expected `m` and `tok` to be globals from gemma_daemon")
    sys.exit(1)


def extract_first_signature(text: str):
    """Find first `def <name>(<args>):` in text, return canonical arg list."""
    m = re.search(r"def\s+(\w+)\s*\(([^)]*)\)\s*:", text)
    if m is None:
        return None
    sig_args = m.group(2).strip()
    if not sig_args:
        return []
    return [canonical_arg(a.strip()) for a in sig_args.split(",") if a.strip()]


n_correct = 0
n_signatures_found = 0
failures = []
for i, p in enumerate(val):
    # Prompt Gemma to emit a function signature
    prompt = (
        f"Problem: {p.question}\n\n"
        f"Write a Python function signature (just the `def name(args):` line) for this problem. "
        f"Do not write the function body.\n\ndef "
    )
    result = m.generate(prompt, tok, max_tokens=40)
    out_text = result["text"] if isinstance(result, dict) else str(result)
    # Gemma continues from "def " — prepend for signature extraction
    full = "def " + out_text
    gemma_args = extract_first_signature(full)
    gt_args = [canonical_arg(a) for a in parse_arg_names(p.expression)]

    if gemma_args is not None:
        n_signatures_found += 1
    match = gemma_args == gt_args
    if match:
        n_correct += 1
    elif i < 10:
        failures.append((p.question[:50], gt_args, gemma_args, out_text[:60]))

    if (i + 1) % 10 == 0:
        print(f"[baseline] {i+1}/{len(val)} done, "
              f"correct so far: {n_correct}")

print(f"\n=== Gemma signature baseline ===")
print(f"  n = {len(val)}")
print(f"  signatures extractable: {n_signatures_found}/{len(val)} = {n_signatures_found/len(val):.2%}")
print(f"  exact arg-list match:   {n_correct}/{len(val)} = {n_correct/len(val):.2%}")
print()
print("Sample failures:")
for q, gt, gm, out in failures[:8]:
    print(f"  prompt: {q!r}")
    print(f"  gt:     {gt}")
    print(f"  gemma:  {gm}")
    print(f"  raw:    {out!r}")
    print()

print("DONE")
