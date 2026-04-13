"""
CALM Matrix backend — determinant, multiply, transpose, inverse.

Models transpose wrong indices and mess up determinants. Pure math.
"""

from __future__ import annotations


def matrix_add(a: list, b: list) -> list:
    """Add two matrices."""
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def matrix_multiply(a: list, b: list) -> list:
    """Multiply two matrices."""
    rows_a, cols_a = len(a), len(a[0])
    cols_b = len(b[0])
    result = [[0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]
    return result


def matrix_transpose(m: list) -> list:
    """Transpose a matrix."""
    return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]


def matrix_determinant(m: list) -> float:
    """Determinant of a square matrix (up to 4x4)."""
    n = len(m)
    if n == 1:
        return float(m[0][0])
    if n == 2:
        return float(m[0][0] * m[1][1] - m[0][1] * m[1][0])
    if n == 3:
        return float(
            m[0][0] * (m[1][1]*m[2][2] - m[1][2]*m[2][1]) -
            m[0][1] * (m[1][0]*m[2][2] - m[1][2]*m[2][0]) +
            m[0][2] * (m[1][0]*m[2][1] - m[1][1]*m[2][0])
        )
    # General case via cofactor expansion
    det = 0
    for j in range(n):
        minor = [row[:j] + row[j+1:] for row in m[1:]]
        det += ((-1) ** j) * m[0][j] * matrix_determinant(minor)
    return float(det)


def matrix_inverse_2x2(m: list) -> list:
    """Inverse of a 2x2 matrix."""
    det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
    if det == 0:
        return [["singular"]]
    inv_det = 1 / det
    return [
        [round(m[1][1] * inv_det, 10), round(-m[0][1] * inv_det, 10)],
        [round(-m[1][0] * inv_det, 10), round(m[0][0] * inv_det, 10)],
    ]


def matrix_trace(m: list) -> float:
    """Trace of a square matrix (sum of diagonal)."""
    return float(sum(m[i][i] for i in range(min(len(m), len(m[0])))))


def matrix_is_symmetric(m: list) -> bool:
    """Check if a matrix is symmetric."""
    n = len(m)
    for i in range(n):
        for j in range(i + 1, n):
            if m[i][j] != m[j][i]:
                return False
    return True


def matrix_scalar_multiply(m: list, scalar: float) -> list:
    """Multiply a matrix by a scalar."""
    s = float(scalar)
    return [[m[i][j] * s for j in range(len(m[0]))] for i in range(len(m))]


def matrix_identity(n: int) -> list:
    """Create an n×n identity matrix."""
    n = int(n)
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


def dot_product(a: list, b: list) -> float:
    """Dot product of two vectors."""
    return float(sum(x * y for x, y in zip(a, b)))


def cross_product(a: list, b: list) -> list:
    """Cross product of two 3D vectors."""
    if len(a) != 3 or len(b) != 3:
        return [0, 0, 0]
    return [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ]


MATRIX_FUNCTIONS = {
    "matrix_add": matrix_add,
    "matrix_multiply": matrix_multiply,
    "matrix_transpose": matrix_transpose,
    "matrix_determinant": matrix_determinant,
    "matrix_inverse_2x2": matrix_inverse_2x2,
    "matrix_trace": matrix_trace,
    "matrix_is_symmetric": matrix_is_symmetric,
    "matrix_scalar_multiply": matrix_scalar_multiply,
    "matrix_identity": matrix_identity,
    "dot_product": dot_product,
    "cross_product": cross_product,
}

MATRIX_NL_PATTERNS = [
    (r'determinant\s+of\s+\[\[(.+?)\]\]', 'matrix_determinant([[{0}]])'),
    (r'transpose\s+of\s+\[\[(.+?)\]\]', 'matrix_transpose([[{0}]])'),
    (r'dot\s+product\s+of\s+\[(.+?)\]\s+and\s+\[(.+?)\]', 'dot_product([{0}], [{1}])'),
    (r'cross\s+product\s+of\s+\[(.+?)\]\s+and\s+\[(.+?)\]', 'cross_product([{0}], [{1}])'),
    (r'(\d+)\s*x\s*\1\s+identity\s+matrix', 'matrix_identity({0})'),
]
