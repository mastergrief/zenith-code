"""
CALM Boolean/binary operations backend — truth values, gates, expressions.

Models botch boolean algebra simplification. Pure computation.
"""

from __future__ import annotations


def and_gate(*args) -> bool:
    """AND gate: True only if all inputs are True."""
    return all(bool(a) for a in args)


def or_gate(*args) -> bool:
    """OR gate: True if any input is True."""
    return any(bool(a) for a in args)


def not_gate(a) -> bool:
    """NOT gate: invert boolean."""
    return not bool(a)


def nand_gate(*args) -> bool:
    """NAND gate: NOT AND."""
    return not all(bool(a) for a in args)


def nor_gate(*args) -> bool:
    """NOR gate: NOT OR."""
    return not any(bool(a) for a in args)


def xor_gate(a, b) -> bool:
    """XOR gate: True if exactly one input is True."""
    return bool(a) != bool(b)


def xnor_gate(a, b) -> bool:
    """XNOR gate: True if both inputs are the same."""
    return bool(a) == bool(b)


def mux(a, b, sel) -> bool:
    """Multiplexer: sel=0 → a, sel=1 → b."""
    return bool(b) if bool(sel) else bool(a)


def half_adder(a, b) -> dict:
    """Half adder: returns {sum, carry}."""
    return {"sum": bool(a) != bool(b), "carry": bool(a) and bool(b)}


def full_adder(a, b, carry_in) -> dict:
    """Full adder: returns {sum, carry_out}."""
    s1 = bool(a) != bool(b)
    c1 = bool(a) and bool(b)
    s2 = s1 != bool(carry_in)
    c2 = s1 and bool(carry_in)
    return {"sum": s2, "carry_out": c1 or c2}


def gray_code(n: int) -> list:
    """Generate n-bit Gray code sequence."""
    n = int(n)
    if n <= 0:
        return [0]
    result = [0, 1]
    for i in range(1, n):
        result = result + [x + (1 << i) for x in reversed(result)]
    return result


def binary_to_gray(n: int) -> int:
    """Convert binary number to Gray code."""
    n = int(n)
    return n ^ (n >> 1)


def gray_to_binary(n: int) -> int:
    """Convert Gray code to binary number."""
    n = int(n)
    result = n
    while n > 0:
        n >>= 1
        result ^= n
    return result


def count_ones(n: int) -> int:
    """Count number of 1-bits (Hamming weight / popcount)."""
    return bin(int(n)).count('1')


def count_zeros(n: int, bits: int = 8) -> int:
    """Count number of 0-bits in an n-bit representation."""
    return int(bits) - count_ones(n)


def parity(n: int) -> int:
    """Parity bit: 0 if even number of 1s, 1 if odd."""
    return count_ones(n) % 2


def hamming_weight(n: int) -> int:
    """Hamming weight (same as popcount/count_ones)."""
    return count_ones(n)


BOOLEAN_FUNCTIONS = {
    "and_gate": and_gate,
    "or_gate": or_gate,
    "not_gate": not_gate,
    "nand_gate": nand_gate,
    "nor_gate": nor_gate,
    "xor_gate": xor_gate,
    "xnor_gate": xnor_gate,
    "mux": mux,
    "half_adder": half_adder,
    "full_adder": full_adder,
    "gray_code": gray_code,
    "binary_to_gray": binary_to_gray,
    "gray_to_binary": gray_to_binary,
    "count_ones": count_ones,
    "count_zeros": count_zeros,
    "parity": parity,
    "hamming_weight": hamming_weight,
}

BOOLEAN_NL_PATTERNS = [
    (r'(?:what is)\s+(\d+)\s+(?:AND|and)\s+(\d+)\s+(?:in binary|boolean)', None),
    (r'(?:gray code|Gray code)\s+(?:for|of)\s+(\d+)\s+bits?', 'gray_code({0})'),
    (r'(?:parity|parity bit)\s+(?:of|for)\s+(\d+)', 'parity({0})'),
    (r'(?:hamming weight|popcount|count ones)\s+(?:of|for|in)\s+(\d+)', 'hamming_weight({0})'),
    (r'half\s+adder\s+(\d)\s+(\d)', 'half_adder({0}, {1})'),
    (r'full\s+adder\s+(\d)\s+(\d)\s+(\d)', 'full_adder({0}, {1}, {2})'),
]
