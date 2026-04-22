"""Code-PT+Delta training data — (problem_description, def_skeleton) pairs
mined from CodeExampleDB.

Produces the training corpus for a `CopyAugmentedDeltaNet` checkpoint that
transduces NL problem descriptions into Python function skeletons:

    "Write a function to convert degrees to radians"
    →
    "def convert_degrees_to_radians(deg):"

Vocab extends math-PT's alphanumeric vocab with code-specific chars:
`:` (colon — required for function headers). Otherwise compatible with
the existing `CopyAugmentedDeltaNet` training pipeline.

Data sources (from CodeExampleDB.examples, deduped on problem hash):
  - mbpp.jsonl         (~970 examples)
  - humanevalplus.jsonl (~164)
  - bigcodebench.jsonl  (~1139)
  - generated/*.jsonl   (~170+)

Extraction: parse solution for `def <name>(<args>):` pattern, prefer the
last top-level def (MBPP pattern: helper classes first, target fn last).
Filter to skeletons ≤ 80 chars + ASCII subset matching vocab.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset


# Extended code vocab: math-PT chars + `:` for function headers
_CODE_CHARS = list(
    "0123456789+-*/()=.,:; "
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "_><"
)
_SPECIAL = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<sep>": 3}
_CODE_CHAR_TO_ID = {**_SPECIAL, **{c: i + len(_SPECIAL) for i, c in enumerate(_CODE_CHARS)}}
_CODE_ID_TO_CHAR = {v: k for k, v in _CODE_CHAR_TO_ID.items()}
CODE_VOCAB_SIZE = len(_CODE_CHAR_TO_ID)

_ALLOWED = set(_CODE_CHARS)


def code_tokenize(text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
    """Char-level tokenize; drop chars not in vocab."""
    ids: List[int] = []
    if add_bos:
        ids.append(_CODE_CHAR_TO_ID["<bos>"])
    for c in text:
        if c in _CODE_CHAR_TO_ID:
            ids.append(_CODE_CHAR_TO_ID[c])
    if add_eos:
        ids.append(_CODE_CHAR_TO_ID["<eos>"])
    return ids


def code_detokenize(ids: List[int]) -> str:
    out = []
    for i in ids:
        c = _CODE_ID_TO_CHAR.get(int(i), "")
        if len(c) > 1:   # special tokens
            continue
        out.append(c)
    return "".join(out)


@dataclass
class CodeProblem:
    """One (problem, skeleton) training pair."""
    question: str         # problem description (max 180 chars)
    expression: str       # function skeleton: `def name(args):`
    difficulty: int = 1   # compatibility with other training code


# Source-file filter: function-defining corpora.
# Expanded from the original 4-source list to include claude_reasoning
# and codecontests, targeting 3-5K raw pairs before paraphrase aug.
_TARGET_CORPORA = (
    "mbpp.jsonl",
    "humanevalplus.jsonl",
    "bigcodebench.jsonl",
    "multi_step_code.jsonl",
    "/generated/",
    "codecontests.jsonl",
    "claude_reasoning.jsonl",
    "claude_reasoning_prefilter.jsonl",
    "claude_reasoning_hf_raw.jsonl",
    "coding_reasoning_claude.jsonl",
    "crownelius.jsonl",
    "nohurry_code.jsonl",
    "python.jsonl",
)


_DEF_RE = re.compile(
    r"^\s*def\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*[^:]+)?:",
    re.MULTILINE,
)


def _clean_prob(prob: str, max_len: int = 180) -> Optional[str]:
    """Normalize a problem description to vocab + length bounds."""
    prob = " ".join(prob.split())      # collapse whitespace
    # Drop vocab-foreign chars
    cleaned = "".join(c if c in _ALLOWED else " " for c in prob)
    cleaned = " ".join(cleaned.split())
    if len(cleaned) < 20 or len(cleaned) > max_len:
        return None
    return cleaned


def _extract_skeleton(
    solution: str,
    placeholder: str = "FN",
) -> Optional[Tuple[str, str]]:
    """Find the target function def in solution. Prefer last top-level
    `def ` (not indented) — MBPP pattern is helper classes first, target
    fn last. Returns (fn_name, skeleton) or None.

    Skeleton uses a GENERIC placeholder (default "FN"). Function names
    like `radian_degree` / `is_coprime` aren't copyable from the prompt
    — they require concept→identifier synthesis beyond DT's copy-
    augmented capability. Arg structure IS learnable (arg count +
    standard names like `s`, `n`, `arr`). Caller substitutes the real
    fn_name at install time.
    """
    sol = solution.replace("\r", "")
    matches = list(_DEF_RE.finditer(sol))
    if not matches:
        return None
    # Prefer top-level (match starts at line start without leading ws)
    top = [m for m in matches if m.group(0).startswith("def ")]
    m = top[-1] if top else matches[-1]
    fn_name = m.group(1)
    args = m.group(2).strip()
    skeleton = f"def {placeholder}({args}):"
    if len(skeleton) > 80:
        return None
    # Verify every char is in vocab (no Unicode / escaped chars)
    if not all(c in _ALLOWED for c in skeleton):
        return None
    return fn_name, skeleton


def extract_pairs_from_db(
    db=None,
    min_len: int = 20,
    max_prob_len: int = 220,
    max_skel_len: int = 80,
    augment: bool = False,
    augment_factor: int = 3,
    aug_seed: int = 42,
) -> List[CodeProblem]:
    """Mine (problem, skeleton) pairs from CodeExampleDB.

    With `augment=True`, applies paraphrase augmentation (template
    rotation on the problem prefix). `augment_factor=N` means each
    original pair spawns N paraphrases. A pair where the prompt
    doesn't start with a known paraphrase template yields 1 pair (no
    aug applied). Roughly 2-3× overall multiplier at factor=3.
    """
    if db is None:
        from calm.llm_computer.facades.code_example_db import CodeExampleDB
        db = CodeExampleDB.load_default()

    pairs: List[CodeProblem] = []
    for ex in db.examples:
        src = ex.source or ""
        if not any(t in src for t in _TARGET_CORPORA):
            continue
        prob = _clean_prob(ex.problem, max_len=max_prob_len)
        if prob is None:
            continue
        result = _extract_skeleton(ex.solution)
        if result is None:
            continue
        _, skeleton = result
        if len(skeleton) > max_skel_len:
            continue
        pairs.append(CodeProblem(question=prob, expression=skeleton))

    if augment:
        pairs = _paraphrase_augment(pairs, factor=augment_factor, seed=aug_seed)
    return pairs


# Template rotation for paraphrase augmentation. Maps a canonical prefix
# (that shows up heavily in MBPP/HumanEval corpora) to alternate phrasings.
# The first match wins; the original unchanged prefix is always preserved
# as one of the paraphrases.
_PARAPHRASE_TEMPLATES = [
    # (canonical prefix regex (case-insensitive), list of replacements)
    (r"^write a (?:python )?function to\b",
     [
         "Write a function to",
         "Write a python function to",
         "Create a function to",
         "Create a python function that",
         "Build a function to",
         "Implement a function to",
         "Define a function to",
         "Python function to",
     ]),
    (r"^write a function that\b",
     [
         "Write a function that",
         "Create a function that",
         "Build a function that",
         "Implement a function that",
         "Define a function that",
     ]),
    (r"^given\b",
     [
         "Given",
         "You are given",
         "For a given",
         "Consider",
     ]),
    (r"^check (?:if|whether)\b",
     [
         "Check if",
         "Check whether",
         "Verify whether",
         "Determine if",
         "Test if",
     ]),
    (r"^find\b",
     [
         "Find",
         "Compute",
         "Return",
         "Identify",
     ]),
    (r"^calculate\b",
     [
         "Calculate",
         "Compute",
         "Return the",
         "Find the",
     ]),
    (r"^(?:count|counts)\b",
     [
         "Count",
         "Count the number of",
         "Return the count of",
         "Compute the count of",
     ]),
    (r"^(?:check|checks)\b",
     [
         "Check",
         "Verify",
         "Determine",
         "Test",
     ]),
    (r"^(?:convert|converts)\b",
     [
         "Convert",
         "Transform",
         "Change",
     ]),
    (r"^(?:sort|sorts)\b",
     [
         "Sort",
         "Order",
         "Arrange",
     ]),
    (r"^return\b",
     [
         "Return",
         "Output",
         "Produce",
         "Give back",
     ]),
    (r"^implement\b",
     [
         "Implement",
         "Write a function to implement",
         "Create",
         "Build",
     ]),
    (r"^(?:remove|removes)\b",
     [
         "Remove",
         "Delete",
         "Filter out",
         "Strip",
     ]),
    (r"^(?:merge|merges)\b",
     [
         "Merge",
         "Combine",
         "Join",
         "Concatenate",
     ]),
    (r"^(?:split|splits)\b",
     [
         "Split",
         "Divide",
         "Partition",
         "Break",
     ]),
]


def _paraphrase_augment(
    pairs: List[CodeProblem], factor: int = 3, seed: int = 42,
) -> List[CodeProblem]:
    """Expand pairs via template-prefix rotation. Each original pair
    spawns up to `factor` paraphrases (including the original)."""
    rng = random.Random(seed)
    out: List[CodeProblem] = []
    for p in pairs:
        variants = [p.question]  # always include original
        matched_template: Optional[list] = None
        matched_match: Optional[re.Match] = None
        for prefix_pat, replacements in _PARAPHRASE_TEMPLATES:
            m = re.match(prefix_pat, p.question, re.IGNORECASE)
            if m:
                matched_template = replacements
                matched_match = m
                break
        if matched_template and matched_match:
            # Pick `factor - 1` distinct alternates. Exclude the
            # replacement that (case-insensitively) matches the original.
            orig_prefix_lower = p.question[:matched_match.end()].lower()
            candidates = [
                r for r in matched_template
                if r.lower() != orig_prefix_lower.rstrip()
            ]
            rng.shuffle(candidates)
            for rep in candidates[:factor - 1]:
                variant = rep + p.question[matched_match.end():]
                # Keep within vocab
                variant = "".join(c if c in _ALLOWED else " " for c in variant)
                variant = " ".join(variant.split())
                if 20 <= len(variant) <= 220:
                    variants.append(variant)
        # Emit all variants with same skeleton
        for v in variants:
            out.append(CodeProblem(question=v, expression=p.expression))
    return out


def split_pairs(
    pairs: List[CodeProblem], val_frac: float = 0.1, seed: int = 42,
) -> Tuple[List[CodeProblem], List[CodeProblem]]:
    """Deterministic train/val split on the extracted pair set."""
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    n_val = max(20, int(len(shuffled) * val_frac))
    return shuffled[n_val:], shuffled[:n_val]


class CodePairDataset(Dataset):
    """Torch dataset over CodeProblem pairs. Each sample is
    (input_ids, target_ids, prefix_len) where input = problem + <sep>,
    target = skeleton tokens (after <sep>).

    Left-padding to max_len. Compatible with the
    CopyAugmentedDeltaNet training loop.
    """

    def __init__(self, pairs: List[CodeProblem], max_len: int = 256):
        self.pairs = pairs
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        # Format: <bos> <prob> <sep> <skel> <eos>, right-padded
        prob_ids = code_tokenize(p.question, add_bos=True, add_eos=False)
        skel_ids = code_tokenize(p.expression, add_bos=False, add_eos=True)
        sep = _CODE_CHAR_TO_ID["<sep>"]
        prefix = prob_ids + [sep]
        tokens = prefix + skel_ids
        prefix_len = len(prefix)
        # Truncate / pad to max_len
        if len(tokens) > self.max_len:
            # Truncate from the front of the problem if too long
            overflow = len(tokens) - self.max_len
            prob_ids = prob_ids[:1] + prob_ids[1 + overflow:]
            prefix = prob_ids + [sep]
            tokens = prefix + skel_ids
            prefix_len = len(prefix)
        pad = _CODE_CHAR_TO_ID["<pad>"]
        while len(tokens) < self.max_len:
            tokens.append(pad)
        return (
            torch.tensor(tokens[:-1], dtype=torch.long),  # input
            torch.tensor(tokens[1:], dtype=torch.long),   # target (shifted)
            prefix_len,
        )


def dump_pairs_jsonl(pairs: List[CodeProblem], path: Path) -> None:
    import json
    with Path(path).open("w") as f:
        for p in pairs:
            f.write(json.dumps({"q": p.question, "skel": p.expression}) + "\n")


def load_pairs_jsonl(path: Path) -> List[CodeProblem]:
    import json
    out: List[CodeProblem] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out.append(CodeProblem(question=rec["q"], expression=rec["skel"]))
    return out


_ARG_NORM_RE = re.compile(r"\s*,\s*")


def normalize_skeleton(skel: str) -> str:
    """Canonicalize `def FN(<args>):` by normalizing whitespace in arg
    list. `FN(a, b):` and `FN(a,b):` and `FN( a , b ):` all collapse
    to `FN(a, b):` (single-space post-comma, no surrounding spaces).

    R6 lever: reduces ~367 output classes by merging spacing variants.
    Safe — outputs identical in function, differ only in formatting.
    """
    s = skel.strip()
    m = re.match(r"^(def FN\()(.*?)(\)\s*:)$", s)
    if not m:
        return s  # malformed — leave alone
    prefix, args, suffix = m.groups()
    # Split on any-whitespace-comma-any-whitespace, rejoin with ", "
    pieces = [p.strip() for p in _ARG_NORM_RE.split(args) if p.strip() or args.strip() == ""]
    # Empty args stays ""
    if args.strip() == "":
        return f"{prefix}){s.rstrip()[-1]}"  # preserve ":"
    return f"{prefix}{', '.join(pieces)}):"


def filter_rare_classes(
    pairs: List[CodeProblem], min_count: int = 3,
) -> List[CodeProblem]:
    """Drop pairs whose skeleton class appears fewer than `min_count`
    times in `pairs`. Use ONLY on training data — val keeps full
    distribution for honest eval. Returns new list.

    R6 lever: classes with 1-2 examples can't generalize, they're
    gradient noise. Dropping them simplifies the learning target.
    """
    from collections import Counter
    counts = Counter(p.expression for p in pairs)
    return [p for p in pairs if counts[p.expression] >= min_count]


def arg_count(skeleton: str) -> int:
    """Return the arg count of a `def FN(<args>):` skeleton.
    Returns -1 for malformed inputs (caller can filter).
    0-arg (FN():) returns 0; 1-arg returns 1; varargs count as 1 arg each.

    R7 lever (family split): groups skeletons by arg count so we can
    train separate DTs per family. Addresses the R1 mode-collapse
    finding by reducing the per-family output space.
    """
    m = re.match(r"^def FN\(([^)]*)\)\s*:$", skeleton.strip())
    if not m:
        return -1
    args = m.group(1).strip()
    if not args:
        return 0
    return len(args.split(","))


def family_bucket(skeleton: str) -> str:
    """Map a skeleton to its family bucket for output-family split.

    Buckets (R7 lever):
      - "zero"   → FN() — 0 args
      - "one"    → FN(arg) — 1 arg  (most common family)
      - "two"    → FN(a, b) — 2 args
      - "three_plus" → FN(a, b, c, ...) — 3+ args

    Returns "unknown" for malformed skeletons.
    """
    n = arg_count(skeleton)
    if n < 0:
        return "unknown"
    if n == 0:
        return "zero"
    if n == 1:
        return "one"
    if n == 2:
        return "two"
    return "three_plus"


def split_pairs_by_family(
    pairs: List[CodeProblem],
) -> dict:
    """Group pairs by family bucket. Returns dict family_name → list.

    R7 lever: each family trains a separate DT with its own output
    vocabulary constraint. Session-31 precedent: per-family PTs beat
    one combined PT by +5-15pp on each family.
    """
    from collections import defaultdict
    buckets: dict = defaultdict(list)
    for p in pairs:
        buckets[family_bucket(p.expression)].append(p)
    return dict(buckets)


def build_balanced_sampler_weights(
    pairs: List[CodeProblem],
    strategy: str = "sqrt_inverse",
    cap: Optional[int] = None,
) -> List[float]:
    """Per-pair weights for a torch WeightedRandomSampler, to counter
    Zipf-distributed skeleton classes.

    Round 1 diagnostic showed DT at 0.193 gate collapses to common
    arg names (FN(n) 89%, FN(list1) 83%) and 0/n on rare classes
    (FN(s), FN(self), FN(x), FN(xs)). Failing classes exist in the
    corpus but are drowned — balanced sampling gives rare skeletons
    training signal proportional to their inverse frequency.

    Strategies:
      - "inverse":      w_i = 1 / count(class(p_i))  — aggressive
      - "sqrt_inverse": w_i = 1 / sqrt(count)        — moderate (default)
      - "capped":       w_i = 1 / min(count, cap)    — bounded lift
      - "uniform":      w_i = 1                      — no-op (control)

    sqrt_inverse is the standard imbalanced-classification heuristic:
    gives rare classes more signal without destroying common-class
    frequency priors.
    """
    import math
    from collections import Counter

    counts = Counter(p.expression for p in pairs)

    if strategy == "uniform":
        return [1.0] * len(pairs)
    if strategy == "inverse":
        return [1.0 / counts[p.expression] for p in pairs]
    if strategy == "sqrt_inverse":
        return [1.0 / math.sqrt(counts[p.expression]) for p in pairs]
    if strategy == "capped":
        if cap is None:
            cap = 20
        return [1.0 / min(counts[p.expression], cap) for p in pairs]
    raise ValueError(f"unknown strategy: {strategy!r}")
