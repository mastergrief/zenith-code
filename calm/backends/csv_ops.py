"""
CALM CSV/TSV backend — parse, validate, stats without pandas.

Pure stdlib csv module. Handles CSV and TSV.
"""

from __future__ import annotations

import csv
import io


def csv_parse(text: str, delimiter: str = ",") -> list:
    """Parse CSV/TSV text into list of rows (list of lists)."""
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    return [row for row in reader]


def csv_headers(text: str, delimiter: str = ",") -> list:
    """Extract header row from CSV/TSV."""
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    for row in reader:
        return row
    return []


def csv_row_count(text: str, delimiter: str = ",") -> int:
    """Count data rows (excludes header)."""
    rows = csv_parse(text, delimiter)
    return max(0, len(rows) - 1)


def csv_column_count(text: str, delimiter: str = ",") -> int:
    """Count columns from header row."""
    headers = csv_headers(text, delimiter)
    return len(headers)


def csv_validate(text: str, delimiter: str = ",") -> str:
    """Validate CSV structure. Returns 'valid' or error description."""
    rows = csv_parse(text, delimiter)
    if not rows:
        return "empty"
    header_len = len(rows[0])
    bad = []
    for i, row in enumerate(rows[1:], 2):
        if len(row) != header_len:
            bad.append(f"row {i}: {len(row)} cols (expected {header_len})")
    if bad:
        return f"inconsistent: {'; '.join(bad[:5])}"
    return "valid"


def csv_column_stats(text: str, column: int, delimiter: str = ",") -> str:
    """Stats for a column (0-indexed): count, nulls, numeric detection."""
    rows = csv_parse(text, delimiter)
    if not rows or column >= len(rows[0]):
        return "invalid column"
    header = rows[0][column] if rows else f"col{column}"
    values = [row[column] for row in rows[1:] if column < len(row)]
    total = len(values)
    nulls = sum(1 for v in values if v.strip() == "")
    nums = 0
    for v in values:
        try:
            float(v)
            nums += 1
        except ValueError:
            pass
    return f"{header}: {total} values, {nulls} nulls, {nums} numeric"


def csv_head(text: str, n: int = 5, delimiter: str = ",") -> list:
    """First N rows (including header)."""
    rows = csv_parse(text, delimiter)
    return rows[: int(n)]


def csv_to_tsv(text: str) -> str:
    """Convert CSV to TSV."""
    rows = csv_parse(text, ",")
    out = io.StringIO()
    writer = csv.writer(out, delimiter="\t")
    writer.writerows(rows)
    return out.getvalue()


def tsv_to_csv(text: str) -> str:
    """Convert TSV to CSV."""
    rows = csv_parse(text, "\t")
    out = io.StringIO()
    writer = csv.writer(out, delimiter=",")
    writer.writerows(rows)
    return out.getvalue()


CSV_FUNCTIONS = {
    "csv_parse": csv_parse,
    "csv_headers": csv_headers,
    "csv_row_count": csv_row_count,
    "csv_column_count": csv_column_count,
    "csv_validate": csv_validate,
    "csv_column_stats": csv_column_stats,
    "csv_head": csv_head,
    "csv_to_tsv": csv_to_tsv,
    "tsv_to_csv": tsv_to_csv,
}
