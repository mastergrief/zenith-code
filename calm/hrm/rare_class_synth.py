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
    "idx": "int", "index": "int", "position": "int", "pos": "int",
    "offset": "int", "depth": "int", "level": "int",
    # String / text
    "s": "string", "text": "string", "str": "string",
    "str1": "string", "str2": "string", "string": "string",
    "test_str": "string", "sentence": "string", "word": "string",
    "name": "string", "char": "string", "prefix": "string",
    "suffix": "string", "pattern": "string", "word1": "string", "word2": "string",
    # List / array
    "l": "list", "arr": "list", "array": "list", "xs": "list",
    "nums": "list", "nums1": "list", "nums2": "list",
    "list": "list", "list1": "list", "list2": "list",
    "input_list": "list", "test_list": "list", "data": "list",
    "items": "list", "elements": "list", "seq": "list", "sequence": "list",
    "values": "list", "keys": "list",
    # Pairs / tuples
    "a": "number", "b": "number", "c": "number", "d": "number",
    "x": "number", "y": "number", "z": "number",
    # Domain-specific (R12: dedicated templates below)
    "url": "url", "request": "request", "db": "db",
    "user": "user", "user_id": "user",
    "matrix": "matrix", "grid": "matrix",
    "tree": "tree", "node": "node", "graph": "graph",
    "file": "file", "path": "file", "filepath": "file", "filename": "file",
    "self": "self",
    # Geometry
    "r": "number", "h": "number", "w": "number",
    "radius": "number", "height": "number", "width": "number",
    "side": "number", "base": "number",
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
    # R12: domain-specific verbs — prompts likely to appear in
    # Claude-authored / web-framework corpora.
    "url": [
        "Parse URL {arg} into components.",
        "Validate URL {arg}.",
        "Extract the domain from URL {arg}.",
        "Return the TLD of URL {arg}.",
        "Check if URL {arg} uses HTTPS.",
        "Normalize URL {arg}.",
        "Return the path portion of URL {arg}.",
    ],
    "db": [
        "Connect to database {arg}.",
        "Query records from {arg}.",
        "Count rows in database {arg}.",
        "Close database connection {arg}.",
        "Return the schema of {arg}.",
    ],
    "request": [
        "Authenticate request {arg}.",
        "Parse headers of request {arg}.",
        "Log incoming request {arg}.",
        "Route request {arg} to the correct handler.",
        "Check CSRF token on request {arg}.",
    ],
    "user": [
        "Fetch user {arg} from the database.",
        "Authenticate user {arg}.",
        "Return the role of user {arg}.",
        "Check whether user {arg} has admin access.",
        "Log out user {arg}.",
    ],
    "file": [
        "Read the contents of file {arg}.",
        "Check whether file {arg} exists.",
        "Return the size of file {arg}.",
        "Open file {arg} for reading.",
        "Delete file {arg}.",
        "Return the extension of file {arg}.",
    ],
    "matrix": [
        "Transpose matrix {arg}.",
        "Return the determinant of matrix {arg}.",
        "Check whether matrix {arg} is square.",
        "Return the trace of matrix {arg}.",
        "Flatten matrix {arg} into a list.",
    ],
    "node": [
        "Return the value of tree node {arg}.",
        "Count descendants of node {arg}.",
        "Return the depth of node {arg}.",
        "Return True if node {arg} is a leaf.",
    ],
    "tree": [
        "Traverse tree {arg} in pre-order.",
        "Return the height of tree {arg}.",
        "Count leaves of tree {arg}.",
        "Return True if tree {arg} is balanced.",
    ],
    "graph": [
        "Return the number of nodes in graph {arg}.",
        "Check whether graph {arg} is connected.",
        "Return the adjacency list of graph {arg}.",
        "Detect a cycle in graph {arg}.",
    ],
    "self": [
        "Return the value of self.",
        "Return the string representation of self.",
        "Reset self to its initial state.",
        "Return a copy of self.",
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
    # R10: more pair combinations — these pairs appear in 2-arg rare
    # classes observed in the corpus.
    ("string", "int"): [
        "Return the character at index {b} of string {a}.",
        "Truncate string {a} to {b} characters.",
        "Repeat string {a} {b} times.",
        "Return the first {b} characters of string {a}.",
        "Check whether string {a} has length at least {b}.",
    ],
    ("int", "string"): [
        "Repeat string {b} exactly {a} times.",
        "Pad string {b} with zeros to width {a}.",
        "Return the {a}th occurrence of string {b}.",
    ],
    ("string", "list"): [
        "Join list {b} using separator {a}.",
        "Count occurrences of string {a} in list {b}.",
        "Filter list {b} to entries containing string {a}.",
    ],
    ("list", "string"): [
        "Filter list {a} to entries matching pattern {b}.",
        "Return elements of list {a} containing string {b}.",
        "Split each element of list {a} by separator {b}.",
    ],
    ("int", "number"): [
        "Round number {b} to {a} decimal places.",
        "Return {b} to the power of {a}.",
    ],
    ("number", "int"): [
        "Round number {a} to {b} decimal places.",
        "Raise number {a} to the integer power {b}.",
    ],
}

# R11: triple templates for common 3-arg patterns observed in corpus
_TEMPLATES_BY_TRIPLE: Dict[Tuple[str, str, str], List[str]] = {
    ("number", "number", "number"): [
        "Return the sum of {a}, {b}, and {c}.",
        "Return the maximum of {a}, {b}, and {c}.",
        "Return the minimum of {a}, {b}, and {c}.",
        "Compute the mean of {a}, {b}, and {c}.",
        "Check if {a}, {b}, and {c} can form a triangle.",
    ],
    ("int", "int", "int"): [
        "Return the GCD of integers {a}, {b}, and {c}.",
        "Check if integer {a} is between {b} and {c}.",
        "Return the sum of integers {a}, {b}, and {c}.",
        "Compute (({a} + {b}) * {c}).",
    ],
    ("list", "int", "int"): [
        "Return the slice of list {a} from index {b} to {c}.",
        "Rotate list {a} by {b} positions toward {c}.",
        "Return the top {b} elements of list {a} starting at offset {c}.",
        "Sort list {a} by key between indices {b} and {c}.",
    ],
    ("list", "number", "number"): [
        "Filter list {a} to elements between {b} and {c}.",
        "Scale each element of list {a} by {b} and offset by {c}.",
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
    fmt_arity = 1   # 1 = {arg}; 2 = {a}/{b}; 3 = {a}/{b}/{c}
    if len(args_raw) == 1:
        templates = _TEMPLATES_BY_SEMANTIC.get(semantics[0], [])
        if not templates:
            templates = _TEMPLATES_BY_SEMANTIC["generic"]
        fmt_arity = 1
    elif len(args_raw) == 2:
        key = (semantics[0], semantics[1])
        templates = _TEMPLATES_BY_PAIR.get(key, [])
        fmt_arity = 2
    elif len(args_raw) == 3:
        key3 = (semantics[0], semantics[1], semantics[2])
        templates = _TEMPLATES_BY_TRIPLE.get(key3, [])
        fmt_arity = 3
    else:
        return []  # 4+ args not handled

    if not templates:
        return []

    out: List[CodeProblem] = []
    for _ in range(n):
        tpl = rng.choice(templates)
        if fmt_arity == 1:
            prompt = tpl.format(arg=args_canon[0])
        elif fmt_arity == 2:
            prompt = tpl.format(a=args_canon[0], b=args_canon[1])
        else:
            prompt = tpl.format(a=args_canon[0], b=args_canon[1], c=args_canon[2])
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
            and arg_count(skel) in (1, 2, 3)]

    out: List[CodeProblem] = []
    for skel in rare:
        synthetic = _generate_for_skeleton(skel, target_per_class, rng)
        out.extend(synthetic)
    return out
