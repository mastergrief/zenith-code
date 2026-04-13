"""
CALM SQL backend — verified query analysis and validation.

Models write broken SQL constantly. This backend parses, validates,
and explains queries deterministically without executing them.

Functions: sql_parse, sql_validate, sql_tables, sql_type, sql_risk, sql_format.
"""

from __future__ import annotations

import re
from typing import Dict, List


def sql_parse(query: str) -> dict:
    """Parse a SQL query into its components.
    Example: sql_parse("SELECT name FROM users WHERE age > 18")
    → {type: "SELECT", tables: ["users"], columns: ["name"], has_where: True, ...}"""
    q = query.strip().rstrip(";")
    upper = q.upper()
    result: Dict = {"raw": q[:200], "valid": True, "errors": []}

    # Determine statement type.
    stmt_types = ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER",
                  "DROP", "TRUNCATE", "GRANT", "REVOKE", "WITH", "EXPLAIN"]
    result["type"] = "UNKNOWN"
    for st in stmt_types:
        if upper.startswith(st):
            result["type"] = st
            break
    if upper.startswith("WITH"):
        result["type"] = "SELECT"  # CTE.

    # Extract tables.
    result["tables"] = _extract_tables(q)

    # Extract columns for SELECT.
    result["columns"] = []
    if result["type"] == "SELECT":
        m = re.search(r'(?i)SELECT\s+(.*?)\s+FROM', q, re.DOTALL)
        if m:
            cols = m.group(1)
            if cols.strip() == "*":
                result["columns"] = ["*"]
            else:
                result["columns"] = [c.strip().split()[-1] for c in cols.split(",")]

    # Clauses present.
    result["has_where"] = bool(re.search(r'\bWHERE\b', upper))
    result["has_join"] = bool(re.search(r'\bJOIN\b', upper))
    result["has_group_by"] = bool(re.search(r'\bGROUP\s+BY\b', upper))
    result["has_order_by"] = bool(re.search(r'\bORDER\s+BY\b', upper))
    result["has_limit"] = bool(re.search(r'\bLIMIT\b', upper))
    result["has_subquery"] = "(" in q and re.search(r'\bSELECT\b', q[q.index("("):], re.I) is not None

    return result


def sql_validate(query: str) -> dict:
    """Validate a SQL query for common errors.
    Returns {valid, errors, warnings}."""
    q = query.strip().rstrip(";")
    upper = q.upper()
    errors: List[str] = []
    warnings: List[str] = []

    # Basic syntax checks.
    if not q:
        errors.append("Empty query")
    elif upper.startswith("SELECT") and "FROM" not in upper and "(" not in q:
        # SELECT without FROM is valid for constants (SELECT 1) but warn.
        if not re.match(r'(?i)SELECT\s+[\d\'"*]', q):
            warnings.append("SELECT without FROM clause")

    # Unmatched parens.
    if q.count("(") != q.count(")"):
        errors.append(f"Unmatched parentheses: {q.count('(')} open, {q.count(')')} close")

    # Unmatched quotes (simple check).
    for ch in ["'", '"']:
        # Count unescaped quotes.
        count = len(re.findall(rf"(?<!\\){re.escape(ch)}", q))
        if count % 2 != 0:
            errors.append(f"Unmatched {ch} quote")

    # DELETE/UPDATE without WHERE.
    if upper.startswith("DELETE") and "WHERE" not in upper:
        warnings.append("DELETE without WHERE — will delete ALL rows")
    if upper.startswith("UPDATE") and "WHERE" not in upper:
        warnings.append("UPDATE without WHERE — will update ALL rows")

    # SELECT * in production.
    if re.search(r'\bSELECT\s+\*\s+FROM\b', upper):
        warnings.append("SELECT * — consider specifying columns")

    # Cartesian join (comma-separated tables without WHERE).
    if upper.startswith("SELECT") and "JOIN" not in upper:
        tables = _extract_tables(q)
        if len(tables) > 1 and "WHERE" not in upper:
            warnings.append(f"Possible cartesian join: {len(tables)} tables without WHERE or JOIN")

    # SQL injection markers.
    injection_patterns = [
        r"'\s*OR\s+['\d].*=.*['\d]",  # ' OR '1'='1
        r";\s*(DROP|DELETE|UPDATE|INSERT)\b",  # stacked queries
        r"UNION\s+SELECT.*FROM\s+information_schema",
        r"--\s*$",  # trailing comment (used to truncate)
    ]
    for pat in injection_patterns:
        if re.search(pat, upper):
            warnings.append("Possible SQL injection pattern detected")
            break

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


def sql_tables(query: str) -> list:
    """Extract all table names referenced in a SQL query.
    Example: sql_tables("SELECT u.name FROM users u JOIN orders o ON u.id = o.user_id")
    → ["users", "orders"]"""
    return _extract_tables(query)


def sql_type(query: str) -> str:
    """Return the SQL statement type.
    Example: sql_type("INSERT INTO users VALUES (1, 'Alice')") → "INSERT" """
    upper = query.strip().upper()
    for st in ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE TABLE",
               "CREATE INDEX", "ALTER", "DROP", "TRUNCATE", "GRANT",
               "REVOKE", "WITH", "EXPLAIN"]:
        if upper.startswith(st):
            return st
    return "UNKNOWN"


def sql_risk(query: str) -> dict:
    """Assess the risk level of a SQL query.
    Returns {risk_level, reasons}. Levels: safe, moderate, high, critical."""
    upper = query.strip().upper()
    reasons = []

    critical = [
        (r'\bDROP\s+(TABLE|DATABASE|SCHEMA)\b', "drops table/database"),
        (r'\bTRUNCATE\b', "truncates table"),
        (r'\bDELETE\b(?!.*\bWHERE\b)', "DELETE without WHERE"),
        (r'\bGRANT\s+ALL\b', "grants all privileges"),
    ]
    high = [
        (r'\bUPDATE\b(?!.*\bWHERE\b)', "UPDATE without WHERE"),
        (r'\bALTER\s+TABLE\b.*\bDROP\b', "drops column"),
        (r'\bDELETE\b', "deletes data"),
    ]
    moderate = [
        (r'\bUPDATE\b', "modifies data"),
        (r'\bINSERT\b', "inserts data"),
        (r'\bALTER\b', "alters schema"),
        (r'\bCREATE\b', "creates object"),
    ]

    for pat, reason in critical:
        if re.search(pat, upper):
            reasons.append(f"CRITICAL: {reason}")
    for pat, reason in high:
        if re.search(pat, upper):
            reasons.append(f"HIGH: {reason}")
    for pat, reason in moderate:
        if re.search(pat, upper):
            reasons.append(f"MODERATE: {reason}")

    if any("CRITICAL" in r for r in reasons):
        level = "critical"
    elif any("HIGH" in r for r in reasons):
        level = "high"
    elif any("MODERATE" in r for r in reasons):
        level = "moderate"
    else:
        level = "safe"

    return {"risk_level": level, "reasons": reasons, "query_type": sql_type(query)}


def sql_format(query: str) -> str:
    """Basic SQL formatting — uppercase keywords, newlines before clauses.
    Example: sql_format("select name from users where age>18")
    → "SELECT name\\nFROM users\\nWHERE age>18" """
    q = query.strip()
    keywords = ["SELECT", "FROM", "WHERE", "JOIN", "LEFT JOIN", "RIGHT JOIN",
                "INNER JOIN", "OUTER JOIN", "CROSS JOIN", "ON", "GROUP BY",
                "HAVING", "ORDER BY", "LIMIT", "OFFSET", "UNION", "INSERT INTO",
                "VALUES", "UPDATE", "SET", "DELETE FROM", "CREATE TABLE",
                "ALTER TABLE", "DROP TABLE", "AND", "OR"]
    result = q
    for kw in sorted(keywords, key=len, reverse=True):
        pattern = re.compile(r'\b' + kw.replace(" ", r'\s+') + r'\b', re.IGNORECASE)
        result = pattern.sub(kw, result)
    # Add newlines before major clauses.
    for clause in ["FROM", "WHERE", "JOIN", "LEFT JOIN", "RIGHT JOIN",
                   "INNER JOIN", "GROUP BY", "HAVING", "ORDER BY",
                   "LIMIT", "UNION", "SET", "VALUES"]:
        result = re.sub(r'\s+(' + clause + r')\b', r'\n\1', result)
    return result.strip()


def _extract_tables(query: str) -> List[str]:
    """Extract table names from a SQL query."""
    tables = []
    q = query.strip()

    # FROM clause.
    for m in re.finditer(r'(?i)\bFROM\s+(\w+)', q):
        tables.append(m.group(1))

    # JOIN clause.
    for m in re.finditer(r'(?i)\bJOIN\s+(\w+)', q):
        tables.append(m.group(1))

    # INSERT INTO.
    m = re.match(r'(?i)INSERT\s+INTO\s+(\w+)', q)
    if m:
        tables.append(m.group(1))

    # UPDATE.
    m = re.match(r'(?i)UPDATE\s+(\w+)', q)
    if m:
        tables.append(m.group(1))

    # DELETE FROM.
    m = re.match(r'(?i)DELETE\s+FROM\s+(\w+)', q)
    if m:
        tables.append(m.group(1))

    # Deduplicate preserving order.
    seen = set()
    result = []
    for t in tables:
        tl = t.lower()
        if tl not in seen and tl not in ("select", "set", "where", "and", "or"):
            seen.add(tl)
            result.append(t)
    return result


SQL_FUNCTIONS = {
    "sql_parse": sql_parse,
    "sql_validate": sql_validate,
    "sql_tables": sql_tables,
    "sql_type": sql_type,
    "sql_risk": sql_risk,
    "sql_format": sql_format,
}

SQL_NL_PATTERNS = [
    (r'(?:validate|is valid|check)\s+(?:this\s+)?(?:SQL|sql)\s+(?:query)?', None),
    (r'(?:what tables?|extract tables?)\s+(?:does|in|from)\s+(?:this\s+)?(?:SQL|sql)', None),
    (r'(?:is)\s+(?:this\s+)?(?:SQL|sql)\s+(?:query\s+)?(?:risky|dangerous|safe)', None),
]
