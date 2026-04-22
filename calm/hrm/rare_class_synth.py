"""Programmatic per-rare-class synthesis (R9).

For each rare skeleton (`def FN(s):`, `def FN(xs):`, `def FN(n, k):` etc.)
that has 3-20 natural-corpus examples, generate 20-40 synthetic (prompt,
skeleton) pairs using the arg names as copy targets.

Why R3 balanced sampling isn't enough: even with inverse-frequency
weighting, the model only sees the 4-8 DISTINCT (prompt, skeleton)
pairs for a rare class. The copy mechanism needs variety — seeing
"reverse s" / "validate s" / "compute length of s" with many surface
forms is what teaches "copy s from prompt". R9 programmatically
generates that variety.

The synthetic pairs are added to the training pool (not val — val
stays natural for honest eval).

Approach: infer each arg's semantic type from its name (n→int,
s→string, arr→list, etc.), then template-generate prompts with
semantic-appropriate verbs, using the EXACT arg names from the
skeleton so the copy path has a target to learn.
"""
from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from calm.hrm.code_dt_data import CodeProblem, arg_count


# --- Arg-name → semantic type inference ---

_SEMANTIC_MAP: Dict[str, str] = {
    # Integer / number
    "n": "int", "N": "int", "num": "int", "number": "int",
    "count": "int", "k": "int", "K": "int", "m": "int", "M": "int",
    "i": "int", "j": "int", "size": "int", "length": "int", "len": "int",
    "limit": "int", "bound": "int", "value": "int",
    # String / text
    "s": "string", "text": "string", "str": "string",
    "str1": "string", "str2": "string", "string": "string",
    "test_str": "string", "sentence": "string", "word": "string",
    "name": "string", "char": "string",
    # List / array
    "l": "list", "arr": "list", "array": "list", "xs": "list",
    "nums": "list", "nums1": "list", "nums2": "list",
    "list": "list", "list1": "list", "list2": "list",
    "input_list": "list", "test_list": "list", "data": "list",
    "items": "list", "elements": "list",
    # Pairs / tuples
    "a": "number", "b": "number", "c": "number", "d": "number",
    "x": "number", "y": "number", "z": "number",
    # Domain-specific (used as-is, no special verbs)
    "url": "url", "request": "request", "db": "db",
    "user": "user", "user_id": "user",
    "matrix": "matrix", "grid": "matrix",
    "tree": "tree", "node": "node", "graph": "graph",
    "file": "file", "path": "file",
    "self": "self",
    # Commonly paired
    "r": "number", "h": "number", "w": "number",
}


def infer_semantic(arg_name: str) -> str:
    """Infer the semantic type of an arg from its name. Returns
    'generic' if unknown."""
    # Strip type annotation: "n: int" → "n"
    base = arg_name.split(":")[0].strip()
    # Strip default value: "n=10" → "n"
    base = base.split("=")[0].strip()
    # Strip * / ** prefixes
    base = base.lstrip("*").strip()
    return _SEMANTIC_MAP.get(base, "generic")


def parse_arg_names(skeleton: str) -> List[str]:
    """Extract the arg names from a `def FN(<args>):` skeleton.
    Keeps annotations for reference but returns the raw tokens."""
    m = re.match(r"^def FN\(([^)]*)\)\s*:$", skeleton.strip())
    if not m:
        return []
    args = m.group(1).strip()
    if not args:
        return []
    return [a.strip() for a in args.split(",") if a.strip()]


def canonical_arg(arg_with_annot: str) -> str:
    """Strip annotation + default from 'n: int = 10' → 'n'."""
    base = arg_with_annot.split(":")[0].strip()
    base = base.split("=")[0].strip()
    return base.lstrip("*")


# --- Template library per semantic type ---

_TEMPLATES_BY_SEMANTIC: Dict[str, List[str]] = {
    "int": [
        "Write a function that takes an integer {arg} and returns whether it is prime.",
        "Create a function to compute the factorial of {arg}.",
        "Write a function that checks if {arg} is even.",
        "Return the number of digits in integer {arg}.",
        "Compute the sum of digits of {arg}.",
        "Return True if {arg} is a perfect square.",
        "Check whether {arg} is a power of 2.",
        "Return the binary representation of {arg}.",
        "Compute the fibonacci number at index {arg}.",
        "Return the reverse of integer {arg}.",
        "Check if integer {arg} is a palindrome.",
        "Return the number of prime factors of {arg}.",
        "Compute {arg} choose 2.",
        "Return True if {arg} divides 100.",
    ],
    "string": [
        "Write a function that reverses string {arg}.",
        "Create a function to check whether string {arg} is a palindrome.",
        "Return the length of string {arg}.",
        "Count the number of vowels in string {arg}.",
        "Return True if string {arg} contains only digits.",
        "Uppercase string {arg}.",
        "Return the number of words in string {arg}.",
        "Remove whitespace from string {arg}.",
        "Check if string {arg} is a valid email.",
        "Return the ASCII sum of characters in string {arg}.",
        "Check whether {arg} is an anagram of its reverse.",
        "Validate that string {arg} is alphanumeric.",
        "Return the most common character in string {arg}.",
    ],
    "list": [
        "Write a function to sort list {arg} in ascending order.",
        "Return the sum of elements in list {arg}.",
        "Find the maximum value in list {arg}.",
        "Return the minimum value in list {arg}.",
        "Count distinct elements in list {arg}.",
        "Filter list {arg} to keep only positive numbers.",
        "Return the mean of list {arg}.",
        "Reverse list {arg}.",
        "Return True if list {arg} is sorted.",
        "Find the second-largest element in list {arg}.",
        "Return the cumulative sum of list {arg}.",
        "Remove duplicates from list {arg}, preserving order.",
        "Return the most frequent element in list {arg}.",
    ],
    "number": [
        "Write a function to square the number {arg}.",
        "Return the absolute value of {arg}.",
        "Check if {arg} is positive.",
        "Return the cube of {arg}.",
        "Write a function that returns {arg} rounded to the nearest integer.",
    ],
    "generic": [
        "Write a function that processes {arg}.",
        "Create a function that validates {arg}.",
        "Return a transformed version of {arg}.",
    ],
}

# 2-arg templates by (type_a, type_b) composition
_TEMPLATES_BY_PAIR: Dict[Tuple[str, str], List[str]] = {
    ("int", "int"): [
        "Write a function to compute the GCD of {a} and {b}.",
        "Return the sum of integers {a} and {b}.",
        "Compute {a} raised to the power of {b}.",
        "Return True if {a} is divisible by {b}.",
        "Return the remainder of {a} divided by {b}.",
        "Compute {a} choose {b}.",
        "Return the larger of integers {a} and {b}.",
        "Return the minimum of {a} and {b}.",
        "Compute the LCM of {a} and {b}.",
    ],
    ("list", "list"): [
        "Return elements common to lists {a} and {b}.",
        "Concatenate lists {a} and {b}.",
        "Return the difference between lists {a} and {b}.",
        "Compute the element-wise sum of {a} and {b}.",
        "Return True if lists {a} and {b} have equal content.",
        "Return the dot product of lists {a} and {b}.",
    ],
    ("list", "int"): [
        "Return the first {b} elements of list {a}.",
        "Rotate list {a} by {b} positions.",
        "Check if list {a} contains {b}.",
        "Return the {b}-th element of list {a}.",
        "Count occurrences of {b} in list {a}.",
    ],
    ("string", "string"): [
        "Return True if strings {a} and {b} are anagrams.",
        "Check whether {a} is a substring of {b}.",
        "Concatenate strings {a} and {b}.",
        "Return the longest common prefix of {a} and {b}.",
        "Compute the edit distance between {a} and {b}.",
    ],
    ("number", "number"): [
        "Compute the distance between {a} and {b}.",
        "Return the average of {a} and {b}.",
        "Check whether {a} equals {b}.",
    ],
    ("int", "list"): [
        "Return the top {a} elements of list {b}.",
        "Check if list {b} has at least {a} elements.",
    ],
}


def _generate_for_skeleton(
    skeleton: str, n: int, rng: random.Random,
) -> List[CodeProblem]:
    """Generate n synthetic (prompt, skeleton) pairs for one skeleton."""
    args_raw = parse_arg_names(skeleton)
    if not args_raw:
        return []
    # Canonical arg names (for filling templates)
    args_canon = [canonical_arg(a) for a in args_raw]
    semantics = [infer_semantic(a) for a in args_raw]

    templates: List[str] = []
    use_arg_fmt = True  # True = 1-arg {arg}; False = 2-arg {a}/{b}
    if len(args_raw) == 1:
        templates = _TEMPLATES_BY_SEMANTIC.get(semantics[0], [])
        if not templates:
            templates = _TEMPLATES_BY_SEMANTIC["generic"]
        use_arg_fmt = True
    elif len(args_raw) == 2:
        key = (semantics[0], semantics[1])
        templates = _TEMPLATES_BY_PAIR.get(key, [])
        use_arg_fmt = False
        # No fallback to 1-arg templates — misleading for 2-arg skeletons
        # (would generate "Compute factorial of a" when skel is FN(a,b)).
        # Caller skips this class; R10 can widen the pair-template library.
    else:
        return []  # 0-arg / 3+arg not handled

    if not templates:
        return []

    out: List[CodeProblem] = []
    for _ in range(n):
        tpl = rng.choice(templates)
        if use_arg_fmt:
            prompt = tpl.format(arg=args_canon[0])
        else:
            prompt = tpl.format(a=args_canon[0], b=args_canon[1])
        out.append(CodeProblem(question=prompt, expression=skeleton))
    return out


def synthesize_rare_class_pairs(
    pairs: List[CodeProblem],
    min_count: int = 3,
    max_count: int = 20,
    target_per_class: int = 30,
    seed: int = 42,
) -> List[CodeProblem]:
    """For each skeleton class with count in [min_count, max_count] in
    `pairs`, generate `target_per_class` synthetic pairs. Returns only
    the new synthetic pairs (caller merges with original corpus).

    Only synthesizes for 1-arg and 2-arg skeletons — 3+ args need a
    wider template library (R10 scope) + risk combinatorial prompts.
    """
    rng = random.Random(seed)
    counts = Counter(p.expression for p in pairs)
    rare = [skel for skel, cnt in counts.items()
            if min_count <= cnt <= max_count
            and arg_count(skel) in (1, 2)]

    out: List[CodeProblem] = []
    for skel in rare:
        synthetic = _generate_for_skeleton(skel, target_per_class, rng)
        out.extend(synthetic)
    return out
