"""
CALM v0.1 stack machine — pure Python reference interpreter.

This is the semantic ground truth for the CALM language. The harness
simulates instructions against this exact VM while streaming LLM
output, and the wasm/native/remote backends must produce identical
results for the subset of words they implement. If behaviour between
backends and this file ever disagree, THIS FILE WINS.

Design
------
Concatenative, stack-based, Forth-inspired. Each instruction is a
WORD (identifier) optionally followed by literal ARGS. The VM owns
one operand stack (a Python list), a tiny call stack for
user-defined words, and a dispatch table mapping word names to
Python callables.

A "word" is either:
  * a BUILTIN: implemented in Python by this file
  * a USER-DEFINED word: a list of sub-instructions registered via
    `:` word_def `;`. When called, its body runs in-line.
  * an EXTERNAL word: dispatched through the backend table. The
    backend is responsible for popping its arguments and pushing
    the result. This VM doesn't care HOW the backend computes the
    answer — wasm, native, remote, anything.

Values on the stack are Python ints, floats, strings, or bools.
Type checks happen at operation time (e.g. `add` requires two
numeric values) and raise `CalmRuntimeError` on mismatch. The
harness catches these errors and converts them to `<error>` tags
in the thinking block.

Instruction format
------------------
A program is a sequence of Instruction objects. Each instruction has
a `word` (str) and `args` (list of literal values, already parsed).

For now the parser is also in this file (see `parse_program`) --
it's small enough to keep together and simpler than importing a
separate module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------

Value = Union[int, float, str, bool]


def _is_number(v: Value) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class CalmError(Exception):
    """Base class — raised by both the parser and the runtime."""


class CalmParseError(CalmError):
    pass


class CalmRuntimeError(CalmError):
    pass


# ---------------------------------------------------------------------------
# Instruction AST
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Instruction:
    word: str
    args: tuple = ()
    source_line: int = -1  # 1-indexed; -1 if synthesized

    def __repr__(self) -> str:
        if self.args:
            return f"{self.word} " + " ".join(_format_literal(a) for a in self.args)
        return self.word


def _format_literal(v: Value) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return '"' + v.replace('"', r'\"') + '"'
    return str(v)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_program(source: str) -> List[Instruction]:
    """
    Parse a CALM source string into a list of Instructions.

    Parsing is token-oriented, NOT line-oriented. Whitespace
    (including newlines) separates tokens; multiple instructions can
    share a line. `\\` starts a line comment. Strings are double-
    quoted with `\\"` as the escape.

    Most words take zero literal arguments -- their inputs come from
    the stack. The two exceptions are:

      * `push <lit>`  consumes the next token as a literal value
      * `: <name>`    consumes the next token as the name of the
                      word being defined

    Everything else parses as a zero-arg instruction. Arguments that
    look like identifiers are passed through as Python strings so
    the VM can later decide whether they're names (e.g. for `:`) or
    data.
    """
    tokens = _tokenize_source(source)
    out: List[Instruction] = []
    i = 0
    while i < len(tokens):
        tok, lineno = tokens[i]
        if tok == "push":
            if i + 1 >= len(tokens):
                raise CalmParseError(
                    f"line {lineno}: 'push' needs an argument"
                )
            arg_tok, _ = tokens[i + 1]
            out.append(
                Instruction(
                    word="push",
                    args=(_parse_literal(arg_tok, lineno),),
                    source_line=lineno,
                )
            )
            i += 2
            continue
        if tok == ":":
            if i + 1 >= len(tokens):
                raise CalmParseError(
                    f"line {lineno}: ':' needs a word name"
                )
            name_tok, _ = tokens[i + 1]
            out.append(
                Instruction(
                    word=":",
                    args=(name_tok,),
                    source_line=lineno,
                )
            )
            i += 2
            continue
        out.append(Instruction(word=tok, args=(), source_line=lineno))
        i += 1
    return out


def _tokenize_source(source: str) -> List[tuple]:
    """
    Tokenize an entire CALM source string into a list of
    (token_str, lineno) pairs. Comments and whitespace are skipped;
    quoted strings are a single token with their quotes preserved.
    """
    tokens: List[tuple] = []
    for lineno, raw in enumerate(source.splitlines(), start=1):
        code = raw.split("\\", 1)[0]
        for tok in _tokenize_line(code, lineno):
            tokens.append((tok, lineno))
    return tokens


def _tokenize_line(code: str, lineno: int) -> List[str]:
    """Simple tokenizer: splits on whitespace, respects double-quoted strings."""
    tokens: List[str] = []
    i = 0
    n = len(code)
    while i < n:
        c = code[i]
        if c.isspace():
            i += 1
            continue
        if c == '"':
            # Consume a quoted string (no escapes beyond \").
            j = i + 1
            while j < n:
                if code[j] == '\\' and j + 1 < n:
                    j += 2
                    continue
                if code[j] == '"':
                    break
                j += 1
            if j >= n:
                raise CalmParseError(
                    f"line {lineno}: unterminated string literal"
                )
            tokens.append(code[i : j + 1])
            i = j + 1
            continue
        # Bareword: until next whitespace.
        j = i
        while j < n and not code[j].isspace():
            j += 1
        tokens.append(code[i:j])
        i = j
    return tokens


def _parse_literal(tok: str, lineno: int) -> Value:
    """Parse a single argument token into a Python Value."""
    if len(tok) >= 2 and tok[0] == '"' and tok[-1] == '"':
        return tok[1:-1].replace(r'\"', '"')
    if tok in ("true", "false"):
        return tok == "true"
    # Try int, then float.
    try:
        return int(tok)
    except ValueError:
        pass
    try:
        return float(tok)
    except ValueError:
        pass
    # Bareword -> interpret as an identifier (kept as a string marker).
    # The VM decides whether to look it up as a word or treat it as a name.
    return tok


# ---------------------------------------------------------------------------
# VM state
# ---------------------------------------------------------------------------

@dataclass
class VMState:
    stack: List[Value] = field(default_factory=list)
    user_words: Dict[str, List[Instruction]] = field(default_factory=dict)
    output: List[Value] = field(default_factory=list)
    halted: bool = False
    # Word definition being captured (if inside `:` ... `;`).
    _defining: Optional[str] = None
    _definition_body: List[Instruction] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

Builtin = Callable[[VMState, Instruction], None]

# A backend callable takes (state, args_from_instruction) and must pop
# its inputs from state.stack and push its output(s). Kept separate
# from builtins so the harness can register per-program backends
# without mutating the core table.
Backend = Callable[[VMState, Instruction], None]


class Dispatcher:
    def __init__(self):
        self.builtins: Dict[str, Builtin] = {}
        self.backends: Dict[str, Backend] = {}

    def register_builtin(self, name: str, fn: Builtin) -> None:
        self.builtins[name] = fn

    def register_backend(self, name: str, fn: Backend) -> None:
        self.backends[name] = fn

    def execute(self, state: VMState, instr: Instruction) -> None:
        if state.halted:
            return

        # Word definition capture mode intercepts everything until
        # `;`. The `:` builtin is allowed to reach its handler so
        # that a nested `:` inside a definition is caught as an
        # error (the handler rejects re-entering defining mode).
        if (
            state._defining is not None
            and instr.word not in (";", ":")
        ):
            state._definition_body.append(instr)
            return

        w = instr.word
        if w in self.builtins:
            self.builtins[w](state, instr)
            return
        if w in state.user_words:
            for sub in state.user_words[w]:
                self.execute(state, sub)
                if state.halted:
                    return
            return
        if w in self.backends:
            self.backends[w](state, instr)
            return
        raise CalmRuntimeError(
            f"line {instr.source_line}: unknown word {w!r}"
        )


# ---------------------------------------------------------------------------
# Stack helpers
# ---------------------------------------------------------------------------

def _require(state: VMState, n: int, word: str) -> List[Value]:
    if len(state.stack) < n:
        raise CalmRuntimeError(
            f"stack underflow: {word!r} needs {n} values, have {len(state.stack)}"
        )
    return state.stack[-n:]


def _pop_n(state: VMState, n: int, word: str) -> List[Value]:
    _require(state, n, word)
    out = state.stack[-n:]
    del state.stack[-n:]
    return out


# ---------------------------------------------------------------------------
# Builtin words
# ---------------------------------------------------------------------------

def _b_push(state: VMState, instr: Instruction) -> None:
    if len(instr.args) != 1:
        raise CalmRuntimeError(
            f"line {instr.source_line}: 'push' needs exactly 1 arg, got {len(instr.args)}"
        )
    state.stack.append(instr.args[0])


def _binop(op_name: str, op):
    def _fn(state: VMState, instr: Instruction) -> None:
        a, b = _pop_n(state, 2, op_name)
        if not (_is_number(a) and _is_number(b)):
            raise CalmRuntimeError(
                f"line {instr.source_line}: {op_name!r} needs numeric operands, got {type(a).__name__}, {type(b).__name__}"
            )
        try:
            state.stack.append(op(a, b))
        except ZeroDivisionError:
            raise CalmRuntimeError(
                f"line {instr.source_line}: division by zero"
            )
    return _fn


def _b_neg(state: VMState, instr: Instruction) -> None:
    (a,) = _pop_n(state, 1, "neg")
    if not _is_number(a):
        raise CalmRuntimeError(
            f"line {instr.source_line}: 'neg' needs numeric operand"
        )
    state.stack.append(-a)


def _b_abs(state: VMState, instr: Instruction) -> None:
    (a,) = _pop_n(state, 1, "abs")
    if not _is_number(a):
        raise CalmRuntimeError(
            f"line {instr.source_line}: 'abs' needs numeric operand"
        )
    state.stack.append(abs(a))


def _b_dup(state: VMState, instr: Instruction) -> None:
    (a,) = _require(state, 1, "dup")
    state.stack.append(a)


def _b_drop(state: VMState, instr: Instruction) -> None:
    _pop_n(state, 1, "drop")


def _b_swap(state: VMState, instr: Instruction) -> None:
    a, b = _pop_n(state, 2, "swap")
    state.stack.append(b)
    state.stack.append(a)


def _b_over(state: VMState, instr: Instruction) -> None:
    a, b = _require(state, 2, "over")  # a is top-2, b is top
    state.stack.append(a)


def _b_rot(state: VMState, instr: Instruction) -> None:
    a, b, c = _pop_n(state, 3, "rot")
    # ( a b c -- b c a )
    state.stack.append(b)
    state.stack.append(c)
    state.stack.append(a)


def _cmpop(op_name: str, op):
    def _fn(state: VMState, instr: Instruction) -> None:
        a, b = _pop_n(state, 2, op_name)
        state.stack.append(bool(op(a, b)))
    return _fn


def _b_emit(state: VMState, instr: Instruction) -> None:
    (a,) = _pop_n(state, 1, "emit")
    state.output.append(a)


def _b_halt(state: VMState, instr: Instruction) -> None:
    state.halted = True


def _b_colon(state: VMState, instr: Instruction) -> None:
    if state._defining is not None:
        raise CalmRuntimeError(
            f"line {instr.source_line}: nested ':' not allowed"
        )
    if len(instr.args) < 1 or not isinstance(instr.args[0], str):
        raise CalmRuntimeError(
            f"line {instr.source_line}: ':' needs a word name"
        )
    state._defining = instr.args[0]
    state._definition_body = []


def _b_semicolon(state: VMState, instr: Instruction) -> None:
    if state._defining is None:
        raise CalmRuntimeError(
            f"line {instr.source_line}: ';' with no matching ':'"
        )
    state.user_words[state._defining] = list(state._definition_body)
    state._defining = None
    state._definition_body = []


def default_dispatcher() -> Dispatcher:
    d = Dispatcher()
    d.register_builtin("push", _b_push)
    d.register_builtin("add", _binop("add", lambda a, b: a + b))
    d.register_builtin("sub", _binop("sub", lambda a, b: a - b))
    d.register_builtin("mul", _binop("mul", lambda a, b: a * b))
    d.register_builtin("div", _binop("div", lambda a, b: a / b))
    d.register_builtin("mod", _binop("mod", lambda a, b: a % b))
    d.register_builtin("neg", _b_neg)
    d.register_builtin("abs", _b_abs)
    d.register_builtin("dup", _b_dup)
    d.register_builtin("drop", _b_drop)
    d.register_builtin("swap", _b_swap)
    d.register_builtin("over", _b_over)
    d.register_builtin("rot", _b_rot)
    d.register_builtin("eq", _cmpop("eq", lambda a, b: a == b))
    d.register_builtin("lt", _cmpop("lt", lambda a, b: a < b))
    d.register_builtin("gt", _cmpop("gt", lambda a, b: a > b))
    d.register_builtin("emit", _b_emit)
    d.register_builtin("halt", _b_halt)
    d.register_builtin(":", _b_colon)
    d.register_builtin(";", _b_semicolon)
    return d


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run(source: str, dispatcher: Optional[Dispatcher] = None) -> VMState:
    """
    Parse and execute a CALM program from source. Returns the final
    VMState (stack, output, halted flag). Errors propagate as
    CalmError subclasses.
    """
    if dispatcher is None:
        dispatcher = default_dispatcher()
    instrs = parse_program(source)
    state = VMState()
    for instr in instrs:
        dispatcher.execute(state, instr)
        if state.halted:
            break
    if state._defining is not None:
        raise CalmRuntimeError(
            f"program ended mid-definition of {state._defining!r}"
        )
    return state


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    src = """
    \\ simplest adder: 17 + 23
    push 17
    push 23
    add
    emit
    halt
    """
    s = run(src)
    print("output:", s.output)
    assert s.output == [40], s.output
    print("OK")
