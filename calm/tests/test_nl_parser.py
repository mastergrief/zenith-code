"""Tests for the natural-language CALM parser."""

import pytest
from calm.nl_parser import normalize_calm_line, normalize_calm_block


class TestFunctionCallSyntax:
    def test_sqrt_parens(self):
        assert normalize_calm_line("sqrt(1764)") == "push 1764\nmath.sqrt"

    def test_is_prime_parens(self):
        assert normalize_calm_line("is_prime(391)") == "push 391\nmath.is_prime"

    def test_gcd_parens(self):
        assert normalize_calm_line("gcd(391, 782)") == "push 391\npush 782\nmath.gcd"

    def test_pow_parens(self):
        assert normalize_calm_line("pow(2, 10)") == "push 2\npush 10\nmath.pow"

    def test_add_parens(self):
        assert normalize_calm_line("add(17, 23)") == "push 17\npush 23\nadd"


class TestInfixSyntax:
    def test_addition(self):
        assert normalize_calm_line("17 + 23") == "push 17\npush 23\nadd"

    def test_multiplication(self):
        assert normalize_calm_line("17 * 23") == "push 17\npush 23\nmul"

    def test_subtraction(self):
        assert normalize_calm_line("100 - 37") == "push 100\npush 37\nsub"

    def test_division(self):
        assert normalize_calm_line("144 / 12") == "push 144\npush 12\ndiv"

    def test_modulo(self):
        assert normalize_calm_line("100 % 3") == "push 100\npush 3\nmod"

    def test_negative_numbers(self):
        assert normalize_calm_line("-5 + 3") == "push -5\npush 3\nadd"


class TestNaturalLanguage:
    def test_multiply_by(self):
        assert normalize_calm_line("multiply 17 by 23") == "push 17\npush 23\nmul"

    def test_add_and(self):
        assert normalize_calm_line("add 17 and 23") == "push 17\npush 23\nadd"

    def test_subtract_from(self):
        assert normalize_calm_line("subtract 37 from 100") == "push 100\npush 37\nsub"

    def test_divide_by(self):
        assert normalize_calm_line("divide 144 by 12") == "push 144\npush 12\ndiv"

    def test_sqrt_of(self):
        assert normalize_calm_line("sqrt of 1764") == "push 1764\nmath.sqrt"

    def test_square_root_of(self):
        assert normalize_calm_line("square root of 1764") == "push 1764\nmath.sqrt"

    def test_gcd_of_and(self):
        assert normalize_calm_line("gcd of 391 and 782") == "push 391\npush 782\nmath.gcd"

    def test_is_prime(self):
        assert normalize_calm_line("is 391 prime?") == "push 391\nmath.is_prime"

    def test_check_if_prime(self):
        assert normalize_calm_line("check if 391 is prime") == "push 391\nmath.is_prime"

    def test_factorize(self):
        assert normalize_calm_line("factorize 391") == "push 391\nmath.factorize"

    def test_to_the_power_of(self):
        assert normalize_calm_line("2 to the power of 10") == "push 2\npush 10\nmath.pow"


class TestBareOps:
    def test_bare_sqrt(self):
        assert normalize_calm_line("sqrt 1764") == "push 1764\nmath.sqrt"

    def test_bare_is_prime(self):
        assert normalize_calm_line("is_prime 391") == "push 391\nmath.is_prime"

    def test_bare_gcd(self):
        assert normalize_calm_line("gcd 391 782") == "push 391\npush 782\nmath.gcd"


class TestPassthrough:
    """Standard stack code should NOT be modified."""

    def test_push_passthrough(self):
        assert normalize_calm_line("push 17") is None

    def test_mul_passthrough(self):
        assert normalize_calm_line("mul") is None

    def test_emit_passthrough(self):
        assert normalize_calm_line("emit") is None

    def test_halt_passthrough(self):
        assert normalize_calm_line("halt") is None


class TestBlockNormalization:
    def test_mixed_block(self):
        block = "multiply 17 by 23 -> <pending>\nis 391 prime? -> <pending>\nhalt"
        result = normalize_calm_block(block)
        lines = result.splitlines()
        assert "push 17" in lines[0]
        assert "mul -> <pending>" in result
        assert "math.is_prime -> <pending>" in result
        assert "halt" in lines[-1]

    def test_comments_preserved(self):
        block = "\\ this is a comment\npush 1"
        result = normalize_calm_block(block)
        assert "\\ this is a comment" in result

    def test_standard_code_unchanged(self):
        block = "push 17\npush 23\nmul -> [391]\nemit\nhalt"
        result = normalize_calm_block(block)
        assert result == block
