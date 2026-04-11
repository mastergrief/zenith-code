"""
Unit tests for the CALM v0.1 stack machine.

Covers: arithmetic, comparisons, stack manipulation, user-defined
words, halt, error paths (stack underflow, bad types, div-by-zero,
unknown word, dangling definition).
"""

from __future__ import annotations

import pytest

from calm.stack_vm import (
    CalmParseError,
    CalmRuntimeError,
    parse_program,
    run,
)


# ----- arithmetic ----------------------------------------------------------

def test_add_ints():
    s = run("push 17\npush 23\nadd\nemit")
    assert s.output == [40]
    assert s.stack == []


def test_all_binops():
    s = run("push 10\npush 3\nadd\nemit")    # 13
    assert s.output == [13]
    s = run("push 10\npush 3\nsub\nemit")    # 7
    assert s.output == [7]
    s = run("push 10\npush 3\nmul\nemit")    # 30
    assert s.output == [30]
    s = run("push 10\npush 3\ndiv\nemit")    # 10/3 == 3.333...
    assert s.output[0] == pytest.approx(10 / 3)
    s = run("push 10\npush 3\nmod\nemit")    # 1
    assert s.output == [1]


def test_neg_abs():
    s = run("push 5\nneg\nemit")
    assert s.output == [-5]
    s = run("push -7\nabs\nemit")
    assert s.output == [7]


def test_float_math():
    s = run("push 1.5\npush 2.5\nadd\nemit")
    assert s.output == [pytest.approx(4.0)]


# ----- stack manipulation --------------------------------------------------

def test_dup():
    s = run("push 5\ndup\nadd\nemit")
    assert s.output == [10]


def test_drop():
    s = run("push 1\npush 2\ndrop\nemit")
    assert s.output == [1]


def test_swap():
    s = run("push 1\npush 2\nswap\nemit\nemit")
    assert s.output == [1, 2]


def test_over():
    # ( 1 2 -- 1 2 1 )
    s = run("push 1\npush 2\nover\nemit\nemit\nemit")
    assert s.output == [1, 2, 1]


def test_rot():
    # ( 1 2 3 -- 2 3 1 )
    s = run("push 1\npush 2\npush 3\nrot\nemit\nemit\nemit")
    assert s.output == [1, 3, 2]


# ----- comparisons ---------------------------------------------------------

def test_comparisons():
    s = run("push 5\npush 5\neq\nemit")
    assert s.output == [True]
    s = run("push 5\npush 6\nlt\nemit")
    assert s.output == [True]
    s = run("push 7\npush 6\ngt\nemit")
    assert s.output == [True]


# ----- user-defined words --------------------------------------------------

def test_simple_word_definition():
    src = """
    : square dup mul ;
    push 7
    square
    emit
    """
    s = run(src)
    assert s.output == [49]


def test_nested_word_calls():
    src = """
    : square dup mul ;
    : sum_of_squares square swap square add ;
    push 3
    push 4
    sum_of_squares
    emit
    """
    s = run(src)
    assert s.output == [25]  # 9 + 16


def test_word_def_persists_across_calls():
    src = """
    : triple push 3 mul ;
    push 4
    triple
    emit
    push 5
    triple
    emit
    """
    s = run(src)
    assert s.output == [12, 15]


# ----- halt ---------------------------------------------------------------

def test_halt_stops_execution():
    src = """
    push 1
    emit
    halt
    push 999
    emit
    """
    s = run(src)
    assert s.output == [1]
    assert s.halted is True


# ----- comments -----------------------------------------------------------

def test_comments_stripped():
    src = r"""
    \ this is a comment
    push 1   \ inline comment
    push 2
    add      \ another
    emit
    """
    s = run(src)
    assert s.output == [3]


# ----- error paths --------------------------------------------------------

def test_stack_underflow():
    with pytest.raises(CalmRuntimeError, match="stack underflow"):
        run("add")


def test_div_by_zero():
    with pytest.raises(CalmRuntimeError, match="division by zero"):
        run("push 1\npush 0\ndiv")


def test_type_mismatch_add():
    with pytest.raises(CalmRuntimeError, match="numeric operands"):
        run('push "hello"\npush 1\nadd')


def test_unknown_word():
    with pytest.raises(CalmRuntimeError, match="unknown word"):
        run("push 1\nnotaword")


def test_dangling_definition():
    with pytest.raises(CalmRuntimeError, match="mid-definition"):
        run(": foo push 1")


def test_semicolon_without_colon():
    with pytest.raises(CalmRuntimeError, match="no matching"):
        run(";")


def test_nested_colon_rejected():
    with pytest.raises(CalmRuntimeError, match="nested"):
        run(": foo : bar")


# ----- parser edge cases --------------------------------------------------

def test_parse_string_literal():
    instrs = parse_program('push "hello world"')
    assert len(instrs) == 1
    assert instrs[0].word == "push"
    assert instrs[0].args == ("hello world",)


def test_parse_float_literal():
    instrs = parse_program("push 3.14")
    assert instrs[0].args == (3.14,)


def test_parse_negative_int():
    instrs = parse_program("push -42")
    assert instrs[0].args == (-42,)


def test_parse_bool_literal():
    instrs = parse_program("push true")
    assert instrs[0].args == (True,)


def test_parse_empty_lines_ignored():
    instrs = parse_program("\n\n  push 1  \n\n\npush 2\n")
    assert [i.word for i in instrs] == ["push", "push"]


def test_parse_unterminated_string():
    with pytest.raises(CalmParseError, match="unterminated string"):
        parse_program('push "oops')


# ----- integration: a non-trivial program --------------------------------

def test_fibonacci_via_word_def():
    """
    Compute the 10th Fibonacci number via a user word plus explicit
    iteration via dup/swap/add. Verifies stack manipulation words
    compose correctly for a real algorithm.

    Algorithm: start with ( a b ) = ( 0 1 ), repeat
      ( a b -- b (a+b) )
    which is 'over + swap' in Forth.
    """
    src = """
    : fib_step over add swap ;
    push 0
    push 1
    fib_step
    fib_step
    fib_step
    fib_step
    fib_step
    fib_step
    fib_step
    fib_step
    fib_step
    fib_step
    drop
    emit
    """
    s = run(src)
    # Fibonacci: 0,1,1,2,3,5,8,13,21,34,55. After 10 steps, top-of-stack
    # is fib(11)=89, but we drop it and emit fib(10)=55 ... actually
    # let's just trust Python's reference implementation.
    def fib_ref(n):
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a
    # 10 fib_steps from (0,1) produces (55, 89) on the stack.
    # drop removes 89, emit outputs 55 = fib(10).
    assert s.output == [fib_ref(10)]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
