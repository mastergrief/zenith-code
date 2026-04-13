"""
CALM Checksum backend — Luhn, ISBN, EAN/UPC validation.

Models cannot reliably compute check digits. Pure math.
"""

from __future__ import annotations


def luhn_validate(number: str) -> bool:
    """Validate a number using the Luhn algorithm (credit cards, IMEI, etc.)."""
    digits = [int(d) for d in str(number).strip() if d.isdigit()]
    if len(digits) < 2:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def luhn_check_digit(number: str) -> int:
    """Compute the Luhn check digit for a number (append this digit to make it valid)."""
    digits = [int(d) for d in str(number).strip() if d.isdigit()]
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return (10 - (total % 10)) % 10


def isbn10_validate(isbn: str) -> bool:
    """Validate an ISBN-10."""
    digits = [c for c in str(isbn).strip() if c.isdigit() or c in "xX"]
    if len(digits) != 10:
        return False
    total = 0
    for i, c in enumerate(digits):
        val = 10 if c in "xX" else int(c)
        total += val * (10 - i)
    return total % 11 == 0


def isbn13_validate(isbn: str) -> bool:
    """Validate an ISBN-13."""
    digits = [int(c) for c in str(isbn).strip() if c.isdigit()]
    if len(digits) != 13:
        return False
    total = sum(d * (1 if i % 2 == 0 else 3) for i, d in enumerate(digits))
    return total % 10 == 0


def isbn_check_digit_13(isbn12: str) -> int:
    """Compute ISBN-13 check digit from first 12 digits."""
    digits = [int(c) for c in str(isbn12).strip() if c.isdigit()]
    if len(digits) != 12:
        return -1
    total = sum(d * (1 if i % 2 == 0 else 3) for i, d in enumerate(digits))
    return (10 - (total % 10)) % 10


def ean_validate(code: str) -> bool:
    """Validate EAN-8 or EAN-13 barcode."""
    digits = [int(c) for c in str(code).strip() if c.isdigit()]
    if len(digits) not in (8, 13):
        return False
    total = sum(d * (1 if i % 2 == 0 else 3) for i, d in enumerate(digits))
    return total % 10 == 0


def upc_validate(code: str) -> bool:
    """Validate UPC-A (12-digit) barcode."""
    digits = [int(c) for c in str(code).strip() if c.isdigit()]
    if len(digits) != 12:
        return False
    total = 0
    for i, d in enumerate(digits):
        total += d * (3 if i % 2 == 0 else 1)
    return total % 10 == 0


def checksum_digit_sum(number: str) -> int:
    """Sum of all digits in a number string."""
    return sum(int(c) for c in str(number) if c.isdigit())


CHECKSUM_FUNCTIONS = {
    "luhn_validate": luhn_validate,
    "luhn_check_digit": luhn_check_digit,
    "isbn10_validate": isbn10_validate,
    "isbn13_validate": isbn13_validate,
    "isbn_check_digit_13": isbn_check_digit_13,
    "ean_validate": ean_validate,
    "upc_validate": upc_validate,
    "checksum_digit_sum": checksum_digit_sum,
}
