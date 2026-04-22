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


# Source-file filter: only function-defining corpora
_TARGET_CORPORA = (
    "mbpp.jsonl",
    "humanevalplus.jsonl",
    "bigcodebench.jsonl",
    "multi_step_code.jsonl",
    "/generated/",
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
    max_prob_len: int = 180,
    max_skel_len: int = 80,
) -> List[CodeProblem]:
    """Mine (problem, skeleton) pairs from CodeExampleDB."""
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
    return pairs


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
