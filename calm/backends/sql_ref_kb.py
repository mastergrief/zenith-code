"""
CALM SQL reference knowledge backend — JOIN types, window functions, constraints.

Models confuse LEFT vs RIGHT JOIN, mess up window function syntax,
hallucinate nonexistent SQL features.
"""

from __future__ import annotations

_DATA_VERSION = "2025-01"

_JOIN_TYPES = {
    "inner join": {
        "description": "Returns rows with matching values in both tables",
        "keeps_unmatched": "neither",
        "example": "SELECT * FROM a INNER JOIN b ON a.id = b.a_id",
    },
    "left join": {
        "description": "Returns all rows from left table + matched rows from right",
        "keeps_unmatched": "left",
        "example": "SELECT * FROM a LEFT JOIN b ON a.id = b.a_id",
        "alias": "LEFT OUTER JOIN",
    },
    "right join": {
        "description": "Returns all rows from right table + matched rows from left",
        "keeps_unmatched": "right",
        "example": "SELECT * FROM a RIGHT JOIN b ON a.id = b.a_id",
        "alias": "RIGHT OUTER JOIN",
    },
    "full outer join": {
        "description": "Returns all rows from both tables, NULL where no match",
        "keeps_unmatched": "both",
        "example": "SELECT * FROM a FULL OUTER JOIN b ON a.id = b.a_id",
    },
    "cross join": {
        "description": "Cartesian product — every row from A paired with every row from B",
        "keeps_unmatched": "n/a (all combinations)",
        "example": "SELECT * FROM a CROSS JOIN b",
        "row_count": "rows_a × rows_b",
    },
    "self join": {
        "description": "Table joined to itself using aliases",
        "keeps_unmatched": "depends on join type used",
        "example": "SELECT e.name, m.name FROM employees e JOIN employees m ON e.manager_id = m.id",
    },
    "natural join": {
        "description": "Implicit join on all columns with same name in both tables",
        "keeps_unmatched": "neither",
        "warning": "Fragile — adding a column can silently change join conditions",
    },
}

_WINDOW_FUNCTIONS = {
    "ROW_NUMBER()": {"description": "Sequential integer for each row in partition", "ties": "arbitrary order"},
    "RANK()": {"description": "Rank with gaps for ties (1,2,2,4)", "ties": "same rank, skip next"},
    "DENSE_RANK()": {"description": "Rank without gaps for ties (1,2,2,3)", "ties": "same rank, no skip"},
    "NTILE(n)": {"description": "Divide rows into n roughly equal groups", "ties": "distributed"},
    "LAG(col, n)": {"description": "Value from n rows before current row", "default": "NULL if no row"},
    "LEAD(col, n)": {"description": "Value from n rows after current row", "default": "NULL if no row"},
    "FIRST_VALUE(col)": {"description": "First value in window frame", "frame": "respects ROWS/RANGE"},
    "LAST_VALUE(col)": {"description": "Last value in window frame", "gotcha": "default frame is RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW — add ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING"},
    "NTH_VALUE(col, n)": {"description": "Nth value in window frame", "frame": "respects ROWS/RANGE"},
    "SUM() OVER": {"description": "Running/cumulative sum within partition", "type": "aggregate window"},
    "AVG() OVER": {"description": "Running/cumulative average within partition", "type": "aggregate window"},
    "COUNT() OVER": {"description": "Running/cumulative count within partition", "type": "aggregate window"},
    "PERCENT_RANK()": {"description": "(rank - 1) / (total_rows - 1)", "range": "0 to 1"},
    "CUME_DIST()": {"description": "Cumulative distribution: fraction of rows <= current", "range": "0 to 1"},
}

_CONSTRAINTS = {
    "PRIMARY KEY": {"description": "Unique + NOT NULL identifier", "per_table": "one", "columns": "one or more"},
    "FOREIGN KEY": {"description": "References PRIMARY KEY in another table", "per_table": "multiple", "actions": "CASCADE, SET NULL, SET DEFAULT, RESTRICT, NO ACTION"},
    "UNIQUE": {"description": "All values must be distinct (NULLs may be allowed)", "per_table": "multiple", "nullable": "depends on DBMS"},
    "NOT NULL": {"description": "Column cannot contain NULL", "per_table": "per-column", "type": "column constraint"},
    "CHECK": {"description": "Custom boolean expression must be true", "per_table": "multiple", "example": "CHECK (age >= 0)"},
    "DEFAULT": {"description": "Value used when INSERT omits the column", "per_table": "per-column", "type": "column constraint"},
    "EXCLUSION": {"description": "PostgreSQL-specific: no two rows satisfy comparison", "per_table": "multiple", "dbms": "PostgreSQL only"},
}

_ISOLATION_LEVELS = {
    "READ UNCOMMITTED": {"dirty_read": True, "nonrepeatable_read": True, "phantom_read": True, "description": "Weakest isolation — can see uncommitted changes"},
    "READ COMMITTED": {"dirty_read": False, "nonrepeatable_read": True, "phantom_read": True, "description": "Default in PostgreSQL, Oracle. Sees only committed data."},
    "REPEATABLE READ": {"dirty_read": False, "nonrepeatable_read": False, "phantom_read": True, "description": "Default in MySQL InnoDB. Same row reads are consistent."},
    "SERIALIZABLE": {"dirty_read": False, "nonrepeatable_read": False, "phantom_read": False, "description": "Strongest. Transactions appear to run one at a time."},
}


def join_type(name: str) -> dict:
    """Get details about a SQL JOIN type."""
    key = str(name).lower().strip()
    if not key.endswith("join"):
        key += " join"
    entry = _JOIN_TYPES.get(key)
    if not entry:
        return {"error": f"Unknown join type: {name}", "valid": list(_JOIN_TYPES.keys())}
    return {"type": key.upper(), **entry}


def window_function(name: str) -> dict:
    """Get details about a SQL window function."""
    n = str(name).upper().strip()
    # Try exact match, then partial
    entry = _WINDOW_FUNCTIONS.get(n)
    if not entry:
        for k, v in _WINDOW_FUNCTIONS.items():
            if n in k:
                return {"function": k, **v}
        return {"error": f"Unknown window function: {name}", "valid": list(_WINDOW_FUNCTIONS.keys())}
    return {"function": n, **entry}


def constraint_info(name: str) -> dict:
    """Get details about a SQL constraint type."""
    key = str(name).upper().strip()
    entry = _CONSTRAINTS.get(key)
    if not entry:
        return {"error": f"Unknown constraint: {name}", "valid": list(_CONSTRAINTS.keys())}
    return {"constraint": key, **entry}


def isolation_level(name: str) -> dict:
    """Get details about a SQL transaction isolation level."""
    key = str(name).upper().strip()
    entry = _ISOLATION_LEVELS.get(key)
    if not entry:
        return {"error": f"Unknown level: {name}", "valid": list(_ISOLATION_LEVELS.keys())}
    return {"level": key, **entry}


def rank_vs_dense_rank() -> dict:
    """Explain the difference between RANK() and DENSE_RANK()."""
    return {
        "RANK": "1, 2, 2, 4 — gaps after ties",
        "DENSE_RANK": "1, 2, 2, 3 — no gaps after ties",
        "ROW_NUMBER": "1, 2, 3, 4 — no ties, arbitrary tiebreak",
        "example_data": [85, 90, 90, 95],
        "RANK_result": [4, 2, 2, 1],
        "DENSE_RANK_result": [3, 2, 2, 1],
        "ROW_NUMBER_result": [4, 2, 3, 1],
    }


def list_window_functions() -> list[str]:
    """List all window functions."""
    return list(_WINDOW_FUNCTIONS.keys())


SQL_REF_FUNCTIONS = {
    "join_type": join_type,
    "window_function": window_function,
    "constraint_info": constraint_info,
    "isolation_level": isolation_level,
    "rank_vs_dense_rank": rank_vs_dense_rank,
    "list_window_functions": list_window_functions,
}

SQL_REF_NL_PATTERNS = [
    (r'(?:what is|explain|difference)\s+(?:a\s+)?(?:sql\s+)?(left|right|inner|full|cross|natural|self)\s+join', 'join_type("{0}")'),
    (r'(?:what is|explain)\s+(?:sql\s+)?(row_number|rank|dense_rank|lag|lead|ntile|first_value|last_value)', 'window_function("{0}")'),
    (r'(?:difference between|vs)\s+rank\s+(?:and|vs)\s+dense.rank', 'rank_vs_dense_rank()'),
    (r'(?:what is|explain)\s+(?:sql\s+)?(read uncommitted|read committed|repeatable read|serializable)', 'isolation_level("{0}")'),
]
