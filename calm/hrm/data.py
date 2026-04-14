"""
HRM Training Data Generator — uses CALM backends to generate verified problems.

Generates math problem/solution pairs at varying difficulty levels.
All solutions verified by safe_eval — guaranteed correct training data.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import List, Dict

import torch
from torch.utils.data import Dataset

from calm.expression import safe_eval, ExpressionError


# Character-level tokenizer for math expressions.
# Letters + comma are included so function calls like gcd(48, 180), fibonacci(n),
# factorial(n), is_prime(n), euler_totient(n), digital_root(n) tokenize
# unambiguously. Without letters the model sees e.g. "(5)" for both
# factorial(5) and fibonacci(5), which caps accuracy on the function categories.
# `;` is used as a step-separator in scratchpad traces
# (`step1 ; step2 ; final`). Semicolon is a regular char in the vocab so it
# tokenizes naturally; no special-token treatment needed.
_CHARS = list("0123456789+-*/()=., ;abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
# <call>/<end_call> bracket expressions that should be delegated to a CALM
# backend at inference time (see HRMSeq2SeqReasoner scratchpad mode).
_SPECIAL = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<sep>": 3, "<call>": 4, "<end_call>": 5}
_CHAR_TO_ID = {**_SPECIAL, **{c: i + len(_SPECIAL) for i, c in enumerate(_CHARS)}}
_ID_TO_CHAR = {v: k for k, v in _CHAR_TO_ID.items()}
VOCAB_SIZE = len(_CHAR_TO_ID)


def tokenize(text: str, max_len: int = 64) -> List[int]:
    """Convert math expression to token IDs."""
    ids = [_CHAR_TO_ID["<bos>"]]
    for c in text:
        if c in _CHAR_TO_ID:
            ids.append(_CHAR_TO_ID[c])
    ids.append(_CHAR_TO_ID["<eos>"])
    # Pad
    while len(ids) < max_len:
        ids.append(_CHAR_TO_ID["<pad>"])
    return ids[:max_len]


def tokenize_trace(trace: str) -> List[int]:
    """Tokenize a scratchpad trace.

    Treats `<call>` and `<end_call>` as single special tokens; everything
    else is char-by-char against `_CHAR_TO_ID`. Unknown chars are dropped.
    Does NOT add `<bos>` / `<eos>` — callers control framing.
    """
    ids: List[int] = []
    i = 0
    while i < len(trace):
        if trace.startswith("<call>", i):
            ids.append(_CHAR_TO_ID["<call>"])
            i += len("<call>")
        elif trace.startswith("<end_call>", i):
            ids.append(_CHAR_TO_ID["<end_call>"])
            i += len("<end_call>")
        else:
            c = trace[i]
            if c in _CHAR_TO_ID:
                ids.append(_CHAR_TO_ID[c])
            i += 1
    return ids


def detokenize_trace(ids: List[int]) -> str:
    """Detokenize a trace-token sequence back to a string.

    Special tokens become their tag form (`<call>`, `<end_call>`).
    Skips `<pad>`, `<bos>`, `<eos>`, `<sep>`.
    """
    chars: List[str] = []
    for i in ids:
        if i == _CHAR_TO_ID["<call>"]:
            chars.append("<call>")
        elif i == _CHAR_TO_ID["<end_call>"]:
            chars.append("<end_call>")
        elif i in (_CHAR_TO_ID["<pad>"], _CHAR_TO_ID["<bos>"],
                   _CHAR_TO_ID["<eos>"], _CHAR_TO_ID["<sep>"]):
            continue
        else:
            chars.append(_ID_TO_CHAR.get(i, "?"))
    return "".join(chars)


def detokenize(ids: List[int]) -> str:
    """Convert token IDs back to string."""
    chars = []
    for i in ids:
        if i in (_CHAR_TO_ID["<pad>"], _CHAR_TO_ID["<bos>"]):
            continue
        if i == _CHAR_TO_ID["<eos>"]:
            break
        if i == _CHAR_TO_ID["<sep>"]:
            chars.append("=")
            continue
        c = _ID_TO_CHAR.get(i, "?")
        chars.append(c)
    return "".join(chars).strip()


@dataclass
class MathProblem:
    """A math problem with verified solution (and optional scratchpad trace)."""
    expression: str
    answer: str
    difficulty: int  # 1-5
    trace: str = ""  # step-by-step trace for scratchpad training (optional)


# --- Scratchpad trace helpers ---

_CALL_TAG = "<call>"
_END_CALL_TAG = "<end_call>"

# Function names that delegate to backends at inference time.
_DELEGATED_FUNCS = ("gcd", "factorial", "fibonacci", "is_prime",
                    "euler_totient", "digital_root")

# Matches `fn(a, b, ...)` where fn is a delegated function name.
_FUNC_CALL_RE = re.compile(
    r'(' + '|'.join(_DELEGATED_FUNCS) + r')\s*\(([^()]*)\)'
)

# Matches the leftmost `a * b` inside the current string (signed ints).
_MULT_RE = re.compile(r'(-?\d+)\s*\*\s*(-?\d+)')
# Matches the leftmost `a + b` or `a - b` (signed ints, non-parenthesized).
_ADDSUB_RE = re.compile(r'(-?\d+)\s*([+\-])\s*(-?\d+)')
# Matches the innermost `(expr)` — no nested parens inside.
_PAREN_RE = re.compile(r'\(([^()]+)\)')


def _format_int(n: int) -> str:
    return str(n)


def _trace_expression(expr: str) -> str:
    """Step-by-step reduction of a CALM-safe_eval-able expression.

    Produces a string of the form:
        "expr = reduction1 = reduction2 = ... = final"

    Function calls to `gcd`, `factorial`, `fibonacci`, etc. are wrapped
    in `<call>...<end_call>` markers — at inference time the reasoner
    intercepts these and delegates to `safe_eval`. During training the
    model learns to emit the markers in the right places.

    Reduction rules, applied iteratively until the expression is a
    single literal:
      1. Evaluate any `<call>fn(args)<end_call>` (already done at data-gen
         time: we replace the call with its numeric result).
      2. Evaluate the innermost parenthesized sub-expression.
      3. Evaluate the leftmost multiplication.
      4. Evaluate the leftmost `+`/`-`.
    """
    work = expr
    steps = [work]

    # Step 1: replace all function calls with <call>...<end_call>RESULT, then
    # keep a cleaned version (without markers) for the subsequent reduction
    # passes. The markers live in `steps` so the model sees them during
    # training; the reduction logic sees plain integers.
    any_call = False
    def sub_call(m):
        nonlocal any_call
        any_call = True
        fn = m.group(1)
        args = m.group(2)
        value = safe_eval(f"{fn}({args})")
        if isinstance(value, bool):
            result_str = str(value)  # True/False
        elif isinstance(value, float) and value == int(value):
            result_str = str(int(value))
        else:
            result_str = str(value)
        return f"{_CALL_TAG}{fn}({args}){_END_CALL_TAG}{result_str}"

    with_calls = _FUNC_CALL_RE.sub(sub_call, work)
    if any_call:
        steps.append(with_calls)
        # For the remaining reduction we want only the literal results.
        work = _FUNC_CALL_RE.sub(lambda m: "", with_calls)
        # After sub(), markers + result still exist; strip them for reduction.
        work = _strip_call_markers(with_calls)
    else:
        work = with_calls

    # Iterative reduction with single-step emission. Place-value decomposition
    # for multi-digit multiplications and adds/subs forces the model to learn
    # sub-operations instead of memorizing full-precision products.
    _LITERAL_RE = re.compile(r'^-?\d+$')

    def reduce_once(s: str):
        # 1. Parens: reduce ONE step inside, then emit. When inner collapses
        #    to a single literal, strip the parens in the same step.
        m = _PAREN_RE.search(s)
        if m:
            inner = m.group(1).strip()
            sub = reduce_once(inner)
            if sub is None:
                return s[:m.start()] + inner + s[m.end():]
            if _LITERAL_RE.match(sub.strip()):
                return s[:m.start()] + sub.strip() + s[m.end():]
            return s[:m.start()] + "(" + sub + ")" + s[m.end():]

        # 2. Multiplication: decompose a*b by place value of b when b is
        #    multi-digit with a non-zero ones digit. Otherwise compute.
        m = _MULT_RE.search(s)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if b >= 10 and b % 10 != 0 and a > 0:
                b_high = (b // 10) * 10
                b_low = b % 10
                expansion = f"({a}*{b_high} + {a}*{b_low})"
                return s[:m.start()] + expansion + s[m.end():]
            return s[:m.start()] + _format_int(a * b) + s[m.end():]

        # 3. Addition/subtraction: compute directly (no decomposition).
        #    +/- are cheap enough that adding digit-split steps bloats traces
        #    without clear benefit. Model is expected to learn 2-digit +/-
        #    via training-data volume (~10K samples covers combinatorial
        #    space well enough).
        m = _ADDSUB_RE.search(s)
        if m:
            a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
            return s[:m.start()] + _format_int(a + b if op == '+' else a - b) + s[m.end():]

        return None

    for _ in range(100):  # safety cap (increased; decomp can produce more steps)
        nxt = reduce_once(work)
        if nxt is None or nxt == work:
            break
        work = nxt
        steps.append(work)

    return " = ".join(s.strip() for s in steps)


def _strip_call_markers(s: str) -> str:
    """Remove <call>expr<end_call> markers but keep the result tokens."""
    return re.sub(re.escape(_CALL_TAG) + r'[^<]+' + re.escape(_END_CALL_TAG), "", s)


def _trace_function_only(fn: str, args: str, answer: str) -> str:
    """Trace for a problem that IS just one function call.

    Format: `<call>fn(args)<end_call>answer = answer`
    The trailing ` = answer` keeps the "last number after last =" extraction
    uniform across all trace shapes.
    """
    return f"{_CALL_TAG}{fn}({args}){_END_CALL_TAG}{answer} = {answer}"


class MathDataGenerator:
    """Generate verified math problems using CALM backends."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def generate(self, n: int = 1000, trace: bool = False) -> List[MathProblem]:
        """Generate n verified math problems.

        When `trace=True`, populates each `MathProblem.trace` with a
        step-by-step scratchpad trace (including `<call>/<end_call>`
        markers around delegated function calls).
        """
        problems = []
        per_category = n // 6

        # Category 1: Simple arithmetic (difficulty 1)
        problems.extend(self._arithmetic_simple(per_category))
        # Category 2: Multi-step arithmetic (difficulty 2)
        problems.extend(self._arithmetic_multi(per_category))
        # Category 3: Parenthesized expressions (difficulty 3)
        problems.extend(self._arithmetic_parens(per_category))
        # Category 4: Function calls (difficulty 3)
        problems.extend(self._functions(per_category))
        # Category 5: Number theory (difficulty 4)
        problems.extend(self._number_theory(per_category))
        # Category 6: Mixed (difficulty 5)
        problems.extend(self._mixed(n - len(problems)))

        if trace:
            for p in problems:
                p.trace = self._build_trace(p)

        self._rng.shuffle(problems)
        return problems[:n]

    def _build_trace(self, p: MathProblem) -> str:
        """Dispatch to the right trace builder based on problem shape."""
        expr = p.expression.strip()
        # Pure function-call problem → single-delegation trace.
        m = _FUNC_CALL_RE.fullmatch(expr)
        if m:
            return _trace_function_only(m.group(1), m.group(2), p.answer)
        # Otherwise fall through to general expression reducer.
        return _trace_expression(expr)

    def _arithmetic_simple(self, n: int) -> List[MathProblem]:
        """a op b"""
        problems = []
        ops = ["+", "-", "*"]
        for _ in range(n):
            a = self._rng.randint(1, 999)
            b = self._rng.randint(1, 999)
            op = self._rng.choice(ops)
            expr = f"{a} {op} {b}"
            try:
                answer = safe_eval(expr)
                problems.append(MathProblem(expr, str(int(answer) if answer == int(answer) else answer), 1))
            except ExpressionError:
                pass
        return problems

    def _arithmetic_multi(self, n: int) -> List[MathProblem]:
        """a op b op c"""
        problems = []
        ops = ["+", "-", "*"]
        for _ in range(n):
            terms = self._rng.randint(3, 4)
            parts = [str(self._rng.randint(1, 50))]
            for _ in range(terms - 1):
                parts.append(self._rng.choice(ops))
                parts.append(str(self._rng.randint(1, 50)))
            expr = " ".join(parts)
            try:
                answer = safe_eval(expr)
                if abs(answer) < 1e8:
                    problems.append(MathProblem(expr, str(int(answer) if float(answer) == int(answer) else round(answer, 2)), 2))
            except (ExpressionError, OverflowError):
                pass
        return problems

    def _arithmetic_parens(self, n: int) -> List[MathProblem]:
        """(a + b) * c"""
        problems = []
        templates = [
            "({a} + {b}) * {c}",
            "({a} - {b}) * {c}",
            "{a} * ({b} + {c})",
            "{a} * ({b} - {c})",
            "({a} + {b}) * ({c} + {d})",
            "({a} + {b}) * ({c} - {d})",
        ]
        for _ in range(n):
            tmpl = self._rng.choice(templates)
            vals = {k: self._rng.randint(2, 30) for k in "abcd"}
            expr = tmpl.format(**vals)
            try:
                answer = safe_eval(expr)
                if abs(answer) < 1e8:
                    problems.append(MathProblem(expr, str(int(answer)), 3))
            except ExpressionError:
                pass
        return problems

    def _functions(self, n: int) -> List[MathProblem]:
        """gcd(a,b), fibonacci(n), factorial(n)"""
        problems = []
        for _ in range(n):
            choice = self._rng.randint(0, 2)
            if choice == 0:
                a, b = self._rng.randint(2, 999), self._rng.randint(2, 999)
                expr = f"gcd({a}, {b})"
            elif choice == 1:
                k = self._rng.randint(1, 15)
                expr = f"fibonacci({k})"
            else:
                k = self._rng.randint(1, 10)
                expr = f"factorial({k})"
            try:
                answer = safe_eval(expr)
                problems.append(MathProblem(expr, str(int(answer)), 3))
            except ExpressionError:
                pass
        return problems

    def _number_theory(self, n: int) -> List[MathProblem]:
        """is_prime(n), euler_totient(n), digital_root(n)"""
        problems = []
        for _ in range(n):
            choice = self._rng.randint(0, 2)
            if choice == 0:
                k = self._rng.randint(2, 200)
                expr = f"is_prime({k})"
            elif choice == 1:
                k = self._rng.randint(2, 50)
                expr = f"euler_totient({k})"
            else:
                k = self._rng.randint(10, 9999)
                expr = f"digital_root({k})"
            try:
                answer = safe_eval(expr)
                problems.append(MathProblem(expr, str(answer), 4))
            except ExpressionError:
                pass
        return problems

    def _mixed(self, n: int) -> List[MathProblem]:
        """Combine arithmetic + functions"""
        problems = []
        templates = [
            "factorial({a}) + {b} * {c}",
            "gcd({a}, {b}) + {c}",
            "fibonacci({a}) * {b}",
            "{a} * {b} + {c} * {d} - {e}",
        ]
        for _ in range(n):
            tmpl = self._rng.choice(templates)
            vals = {"a": self._rng.randint(2, 8), "b": self._rng.randint(2, 20),
                    "c": self._rng.randint(2, 20), "d": self._rng.randint(2, 15),
                    "e": self._rng.randint(1, 10)}
            expr = tmpl.format(**vals)
            try:
                answer = safe_eval(expr)
                if abs(answer) < 1e8:
                    problems.append(MathProblem(expr, str(int(answer) if float(answer) == int(answer) else round(answer, 4)), 5))
            except (ExpressionError, OverflowError):
                pass
        return problems


_DIGIT_RE = re.compile(r'^-?\d+(?:\.\d+)?$')
# Matches the *reversed* form (trailing sign, embedded decimal at arbitrary
# position, only digit/sign/decimal chars).
_REVERSED_DIGIT_RE = re.compile(r'^[0-9.\-]+$')


def _maybe_reverse_digits(s: str) -> str:
    """Reverse the string iff it looks like a pure number.

    `"1218"` -> `"8121"`, `"3628800"` -> `"0088263"`, `"-45"` -> `"54-"`.
    Leaves non-numeric strings (`"True"`, `"False"`) untouched so the
    model still outputs them in their natural order.

    The decoder predicts ones digit first so carry propagates naturally
    left-to-right during generation (standard transformer-arithmetic
    trick, per the Abacus paper).
    """
    if _DIGIT_RE.match(s.strip()):
        return s[::-1]
    return s


def _unreverse_if_numeric(s: str) -> str:
    """Inverse of `_maybe_reverse_digits` for decoded model output.

    The reversed form of "-985" is "589-" which does NOT match the
    leading-sign regex — so we need a separate detector keyed on the
    reversed-numeric character set (digits + trailing minus + embedded
    decimal) to know when to un-reverse.

    Leaves alphabetic outputs (True/False, garbage) untouched.
    """
    st = s.strip()
    if st and _REVERSED_DIGIT_RE.match(st) and any(c.isdigit() for c in st):
        return s[::-1]
    return s


class MathDataset(Dataset):
    """PyTorch Dataset for HRM training (autoregressive next-token).

    Format: full sequence is <bos> expr = answer <eos> <pad>...
    Input = full[:-1], target = full[1:]. Loss mask covers the positions
    whose predicted next-token falls in the answer region (including <eos>).

    The nested L/H recurrent loops inside the model operate per forward
    pass; causality is enforced by causal self-attention inside each block.
    """

    def __init__(self, problems: List[MathProblem], max_len: int = 64):
        self.problems = problems
        self.max_len = max_len

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        p = self.problems[idx]
        pad_id = _CHAR_TO_ID["<pad>"]
        bos_id = _CHAR_TO_ID["<bos>"]
        eos_id = _CHAR_TO_ID["<eos>"]

        expr_str = p.expression + "="
        expr_ids = [bos_id] + [_CHAR_TO_ID[c] for c in expr_str if c in _CHAR_TO_ID]
        answer_ids = [_CHAR_TO_ID[c] for c in p.answer if c in _CHAR_TO_ID] + [eos_id]

        full = expr_ids + answer_ids
        full = full[: self.max_len]  # truncate pathological cases
        while len(full) < self.max_len:
            full.append(pad_id)

        # Shifted pairs. Keep both length max_len by appending pad to the tail.
        input_ids = full[:-1] + [pad_id]
        target_ids = full[1:] + [pad_id]

        # loss_mask[i] = 1 iff target[i] (= full[i+1]) is an answer/eos token.
        # expr_ids ends at index len(expr_ids)-1, answer tokens occupy
        # full[len(expr_ids) .. len(expr_ids)+len(answer_ids)-1].
        L = len(expr_ids)
        A = len(answer_ids)
        mask = [0] * self.max_len
        for i in range(L - 1, min(L - 1 + A, self.max_len)):
            mask[i] = 1

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "loss_mask": torch.tensor(mask, dtype=torch.bool),
        }


class MathSeq2SeqDataset(Dataset):
    """PyTorch Dataset for HRMSeq2Seq training.

    Encoder input:  <bos> expression <eos> <pad> ...           (length = max_enc_len, bidirectional)
    Decoder input:  <bos> target[:-1]                          (length = max_dec_len)
    Decoder target: target[1:] <eos> <pad> ...                 (length = max_dec_len)
    Loss mask:      1 at positions predicting target/eos, 0 elsewhere.

    Three modes:
      - `use_trace=True`: decoder target is the problem's scratchpad trace
        (with `<call>/<end_call>` markers). Forces `reverse_digits=False`.
        This is the mode for Round 1c (scratchpad + backend-call hybrid).
      - `use_trace=False, reverse_digits=True`: decoder target is the
        raw answer with numeric digits reversed (Round 1a A+B).
      - `use_trace=False, reverse_digits=False`: decoder target is the
        raw answer in natural order (Round 1a ablation).
    """

    def __init__(self, problems: List[MathProblem], max_enc_len: int = 32,
                 max_dec_len: int = 16, reverse_digits: bool = True,
                 use_trace: bool = False, structure_only: bool = False):
        self.problems = problems
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len
        self.use_trace = use_trace
        # Trace mode is incompatible with digit-reversal (intermediate numbers
        # in the trace would get reversed mid-computation, breaking training).
        self.reverse_digits = reverse_digits and not use_trace
        # Structure-only mode: loss is applied only up to the first `=` in
        # the decoder target — i.e. the problem expression + the equals
        # terminator. Everything after `=` (reduction / value computation)
        # gets zero loss weight. Downstream, the `--verified` inference
        # path parses the pre-`=` segment and recomputes values via
        # LLM-Computer's interpreter, so HRM never needs to learn to
        # produce correct values. Works in both trace and answer-only
        # modes — the first-`=` boundary separates structure from value
        # in both formats.
        self.structure_only = structure_only

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        p = self.problems[idx]
        pad_id = _CHAR_TO_ID["<pad>"]
        bos_id = _CHAR_TO_ID["<bos>"]
        eos_id = _CHAR_TO_ID["<eos>"]

        # --- Encoder: <bos> expression <eos> pad... ---
        enc_ids = [bos_id] + [_CHAR_TO_ID[c] for c in p.expression if c in _CHAR_TO_ID] + [eos_id]
        enc_ids = enc_ids[: self.max_enc_len]
        while len(enc_ids) < self.max_enc_len:
            enc_ids.append(pad_id)

        # --- Decoder target: trace / structure-only / raw answer ---
        if self.structure_only:
            # Minimal target: `problem = <eos>`. Model learns to echo the
            # problem expression and emit an `=` terminator. The LLM-Computer
            # parser + interpreter handle every value downstream.
            target_token_ids = tokenize_trace(p.expression + "=") + [eos_id]
        elif self.use_trace and p.trace:
            target_token_ids = tokenize_trace(p.trace) + [eos_id]
        else:
            answer = _maybe_reverse_digits(p.answer) if self.reverse_digits else p.answer
            target_token_ids = [_CHAR_TO_ID[c] for c in answer if c in _CHAR_TO_ID] + [eos_id]

        # Teacher-forced: decoder_in = <bos> target[:-1], decoder_target = target
        dec_in = [bos_id] + target_token_ids[:-1]
        dec_target = target_token_ids
        dec_in = dec_in[: self.max_dec_len]
        dec_target = dec_target[: self.max_dec_len]
        while len(dec_in) < self.max_dec_len:
            dec_in.append(pad_id)
        while len(dec_target) < self.max_dec_len:
            dec_target.append(pad_id)

        # Loss mask: 1 at positions whose target is non-pad (real target tokens
        # including the terminal <eos>).
        mask = [1 if t != pad_id else 0 for t in dec_target]

        # Structure-only targets are already minimal (problem + = + eos),
        # so no further masking is needed — every non-pad position is
        # already a structure token.

        return {
            "encoder_ids": torch.tensor(enc_ids, dtype=torch.long),
            "decoder_input_ids": torch.tensor(dec_in, dtype=torch.long),
            "decoder_target_ids": torch.tensor(dec_target, dtype=torch.long),
            "loss_mask": torch.tensor(mask, dtype=torch.bool),
        }
