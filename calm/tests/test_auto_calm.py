"""Tests for Auto-CALM — transparent claim verification."""

import pytest
from calm.auto_calm import AutoCalm, Claim


class TestClaimExtraction:
    """Test numeric claim extraction from text."""

    def setup_method(self):
        self.ac = AutoCalm()

    def test_basic_arithmetic(self):
        claims = self.ac.extract_claims("17 * 23 = 391")
        assert len(claims) == 1
        assert claims[0].claimed_value == "391"

    def test_latex_times(self):
        claims = self.ac.extract_claims(r"17 \times 23 = 391")
        assert len(claims) == 1
        assert claims[0].expression == "17 * 23"

    def test_unicode_times(self):
        claims = self.ac.extract_claims("17 × 23 = 391")
        assert len(claims) == 1

    def test_no_false_positive_on_equation_fragment(self):
        """391 = 17 × 20 + 51 should NOT match '391 = 17'."""
        claims = self.ac.extract_claims(r"391 = 17 \times 20 + 51")
        assert len(claims) == 0

    def test_no_false_positive_plain_number(self):
        """'391 = 17' should not match — LHS has no operator."""
        claims = self.ac.extract_claims("391 = 17")
        assert len(claims) == 0

    def test_function_call_claim(self):
        claims = self.ac.extract_claims("factorial(10) = 3628800")
        assert len(claims) == 1
        assert claims[0].expression == "factorial(10)"

    def test_gcd_claim(self):
        claims = self.ac.extract_claims("GCD of 391 and 782 is 391")
        assert len(claims) == 1
        assert claims[0].expression == "gcd(391, 782)"

    def test_lcm_claim(self):
        claims = self.ac.extract_claims("LCM of 12 and 8 is 24")
        assert len(claims) == 1
        assert claims[0].expression == "lcm(12, 8)"

    def test_commas_in_numbers(self):
        claims = self.ac.extract_claims(r"100,283 \div 17 = 5,899")
        assert len(claims) == 1
        assert claims[0].claimed_value == "5899"


class TestBoolClaims:
    """Test boolean claim extraction."""

    def setup_method(self):
        self.ac = AutoCalm()

    def test_is_prime(self):
        claims = self.ac.extract_bool_claims("391 is not prime")
        assert len(claims) == 1
        assert claims[0].expression == "is_prime(391)"
        assert claims[0].claimed_value == "False"

    def test_is_prime_positive(self):
        claims = self.ac.extract_bool_claims("7 is prime")
        assert len(claims) == 1
        assert claims[0].claimed_value == "True"

    def test_is_perfect(self):
        claims = self.ac.extract_bool_claims("28 is a perfect number")
        assert len(claims) == 1
        assert claims[0].expression == "is_perfect(28)"

    def test_divisible(self):
        claims = self.ac.extract_bool_claims("1089 is divisible by 3")
        assert len(claims) == 1
        assert claims[0].expression == "1089 % 3 == 0"

    def test_conditional_if_excluded(self):
        """'if X is prime' is a question, not an assertion."""
        claims = self.ac.extract_bool_claims("To determine if 391 is prime")
        assert len(claims) == 0

    def test_conditional_whether_excluded(self):
        claims = self.ac.extract_bool_claims("whether 28 is a perfect number")
        assert len(claims) == 0

    def test_conditional_check_excluded(self):
        claims = self.ac.extract_bool_claims(
            "We check if the digit sum (42) is divisible by 9"
        )
        assert len(claims) == 0

    def test_perfectly_not_perfect(self):
        """'perfectly divisible' should NOT match 'is perfect'."""
        claims = self.ac.extract_bool_claims(
            "100283 is perfectly divisible by 17"
        )
        assert len(claims) == 0

    def test_parens_around_number(self):
        claims = self.ac.extract_bool_claims("(391) is not prime")
        assert len(claims) == 1


class TestVerification:
    """Test claim verification and correction."""

    def setup_method(self):
        self.ac = AutoCalm()

    def test_correct_arithmetic(self):
        _, report = self.ac.verify_and_correct("17 * 23 = 391")
        assert report.corrections == 0
        assert report.verified == 1

    def test_wrong_arithmetic_corrected(self):
        corrected, report = self.ac.verify_and_correct("17 * 23 = 401")
        assert report.corrections == 1
        assert "391" in corrected

    def test_correct_bool(self):
        _, report = self.ac.verify_and_correct("391 is not prime")
        assert report.corrections == 0
        assert report.verified == 1

    def test_wrong_bool_corrected(self):
        corrected, report = self.ac.verify_and_correct("391 is prime")
        assert report.corrections == 1
        assert "not prime" in corrected

    def test_wrong_gcd_corrected(self):
        corrected, report = self.ac.verify_and_correct(
            "The GCD of 391 and 782 is 17"
        )
        assert report.corrections == 1
        assert "391" in corrected

    def test_integer_division_with_remainder(self):
        """54 ÷ 7 = 7 remainder 5 is correct (integer division)."""
        _, report = self.ac.verify_and_correct(
            r"54 \div 7 = 7 remainder 5"
        )
        assert report.corrections == 0

    def test_no_claims_no_change(self):
        text = "The weather is nice today."
        corrected, report = self.ac.verify_and_correct(text)
        assert corrected == text
        assert len(report.claims) == 0

    def test_latex_stripped(self):
        """LaTeX formatting should be stripped for extraction."""
        _, report = self.ac.verify_and_correct(
            r"$17 \times 23 = \mathbf{391}$"
        )
        assert report.verified >= 1

    def test_markdown_stripped(self):
        """Markdown bold should be stripped for extraction."""
        _, report = self.ac.verify_and_correct("17 * 23 = **391**")
        assert report.verified >= 1

    def test_multiple_claims(self):
        text = "17 * 23 = 391 and 42 * 19 = 798"
        _, report = self.ac.verify_and_correct(text)
        assert len(report.claims) == 2
        assert report.verified == 2


class TestPrecompute:
    """Test precomputation extraction from prompts."""

    def setup_method(self):
        self.engine = __import__("calm.auto_calm", fromlist=["AutoCalmEngine"]).AutoCalmEngine()

    def test_arithmetic(self):
        r = self.engine._precompute("What is 17 * 23?")
        assert "17 * 23" in r
        assert r["17 * 23"] == 391

    def test_factorial(self):
        r = self.engine._precompute("What is factorial(10)?")
        assert "factorial(10)" in r
        assert r["factorial(10)"] == 3628800

    def test_fibonacci(self):
        r = self.engine._precompute("What is fibonacci(20)?")
        assert "fibonacci(20)" in r
        assert r["fibonacci(20)"] == 6765

    def test_gcd(self):
        r = self.engine._precompute("What is the GCD of 391 and 782?")
        assert "gcd(391, 782)" in r
        assert r["gcd(391, 782)"] == 391

    def test_collatz_length(self):
        r = self.engine._precompute(
            "How long is the Collatz sequence starting from 27?"
        )
        assert "collatz_length(27)" in r
        assert r["collatz_length(27)"] == 112

    def test_is_prime(self):
        r = self.engine._precompute("Is 1000003 prime?")
        assert "is_prime(1000003)" in r
        assert r["is_prime(1000003)"] is True

    def test_is_perfect(self):
        r = self.engine._precompute("Is 28 a perfect number?")
        assert "is_perfect(28)" in r
        assert r["is_perfect(28)"] is True

    def test_no_precompute_for_prose(self):
        r = self.engine._precompute("Tell me about prime numbers.")
        assert len(r) == 0
