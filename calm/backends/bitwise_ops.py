"""
CALM bitwise backend — verified bit manipulation operations.

Pure computation the model can't do mentally. Masks, shifts, flags,
bit counting, two's complement — all deterministic.

Functions: bit_and, bit_or, bit_xor, bit_not, bit_shift, bit_count,
popcount, leading_zeros, trailing_zeros, bit_set, bit_clear, bit_test,
to_binary, from_binary, twos_complement.
"""

from __future__ import annotations


def bit_and(a: int, b: int) -> int:
    """Bitwise AND. bit_and(0xFF, 0x0F) → 15"""
    return int(a) & int(b)


def bit_or(a: int, b: int) -> int:
    """Bitwise OR. bit_or(0xF0, 0x0F) → 255"""
    return int(a) | int(b)


def bit_xor(a: int, b: int) -> int:
    """Bitwise XOR. bit_xor(0xFF, 0x0F) → 240"""
    return int(a) ^ int(b)


def bit_not(a: int, bits: int = 32) -> int:
    """Bitwise NOT with specified width. bit_not(0, 8) → 255"""
    bits = int(bits)
    mask = (1 << bits) - 1
    return int(a) ^ mask


def bit_shift_left(a: int, n: int) -> int:
    """Left shift. bit_shift_left(1, 4) → 16"""
    n = min(int(n), 64)  # Cap shift amount.
    return int(a) << n


def bit_shift_right(a: int, n: int) -> int:
    """Right shift (logical). bit_shift_right(16, 4) → 1"""
    n = min(int(n), 64)
    return int(a) >> n


def popcount(n: int) -> int:
    """Count the number of set bits (1s). popcount(255) → 8"""
    return bin(int(n)).count("1")


def leading_zeros(n: int, bits: int = 32) -> int:
    """Count leading zeros in a fixed-width representation.
    leading_zeros(1, 8) → 7"""
    n, bits = int(n), int(bits)
    if n <= 0:
        return bits if n == 0 else 0
    return bits - n.bit_length()


def trailing_zeros(n: int) -> int:
    """Count trailing zeros. trailing_zeros(8) → 3 (binary: 1000)"""
    n = int(n)
    if n == 0:
        return 0
    count = 0
    while (n & 1) == 0:
        count += 1
        n >>= 1
    return count


def bit_set(n: int, pos: int) -> int:
    """Set bit at position. bit_set(0, 3) → 8"""
    return int(n) | (1 << int(pos))


def bit_clear(n: int, pos: int) -> int:
    """Clear bit at position. bit_clear(15, 3) → 7"""
    return int(n) & ~(1 << int(pos))


def bit_test(n: int, pos: int) -> bool:
    """Test if bit at position is set. bit_test(8, 3) → True"""
    return bool(int(n) & (1 << int(pos)))


def to_binary(n: int, bits: int = 0) -> str:
    """Convert integer to binary string. to_binary(42, 8) → "00101010" """
    n = int(n)
    bits = int(bits)
    if bits > 0:
        return format(n & ((1 << bits) - 1), f"0{bits}b")
    return bin(n)[2:] if n >= 0 else "-" + bin(n)[3:]


def from_binary(s: str) -> int:
    """Convert binary string to integer. from_binary("00101010") → 42"""
    return int(str(s).strip(), 2)


def twos_complement(n: int, bits: int = 32) -> int:
    """Get the two's complement representation.
    twos_complement(-1, 8) → 255, twos_complement(255, 8) → -1"""
    n, bits = int(n), int(bits)
    mask = (1 << bits) - 1
    if n < 0:
        return n & mask  # Negative → unsigned.
    elif n >= (1 << (bits - 1)):
        return n - (1 << bits)  # Unsigned → signed negative.
    return n


def bit_reverse(n: int, bits: int = 32) -> int:
    """Reverse bits. bit_reverse(1, 8) → 128"""
    n, bits = int(n), int(bits)
    result = 0
    for _ in range(bits):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result


def byte_swap(n: int, size: int = 4) -> int:
    """Swap byte order (endianness). byte_swap(0x12345678, 4) → 0x78563412"""
    n, size = int(n), int(size)
    b = n.to_bytes(size, byteorder="big", signed=n < 0)
    return int.from_bytes(b, byteorder="little", signed=False)


def mask(start: int, end: int) -> int:
    """Create a bitmask from start to end (inclusive).
    mask(2, 5) → 0b111100 = 60"""
    start, end = int(start), int(end)
    if start > end:
        start, end = end, start
    return ((1 << (end - start + 1)) - 1) << start


BITWISE_FUNCTIONS = {
    "bit_and": bit_and,
    "bit_or": bit_or,
    "bit_xor": bit_xor,
    "bit_not": bit_not,
    "bit_shift_left": bit_shift_left,
    "bit_shift_right": bit_shift_right,
    "popcount": popcount,
    "leading_zeros": leading_zeros,
    "trailing_zeros": trailing_zeros,
    "bit_set": bit_set,
    "bit_clear": bit_clear,
    "bit_test": bit_test,
    "to_binary": to_binary,
    "from_binary": from_binary,
    "twos_complement": twos_complement,
    "bit_reverse": bit_reverse,
    "byte_swap": byte_swap,
    "mask": mask,
}

BITWISE_NL_PATTERNS = [
    (r'(\d+)\s+(?:AND|and|&)\s+(\d+)\s+(?:bitwise|in binary)', 'bit_and({0}, {1})'),
    (r'(\d+)\s+(?:OR|or|\|)\s+(\d+)\s+(?:bitwise|in binary)', 'bit_or({0}, {1})'),
    (r'(\d+)\s+(?:XOR|xor|\^)\s+(\d+)\s+(?:bitwise|in binary)', 'bit_xor({0}, {1})'),
    (r'popcount\s+(?:of\s+)?(\d+)', 'popcount({0})'),
    (r'(?:number of|count|how many)\s+(?:set\s+)?bits?\s+in\s+(\d+)', 'popcount({0})'),
    (r'twos?\s+complement\s+(?:of\s+)?(-?\d+)', 'twos_complement({0})'),
]
