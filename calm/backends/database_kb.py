"""
CALM Database knowledge backend — ACID, CAP, normalization, indexing.

Models confuse ACID properties, mix up normal forms, hallucinate CAP theorem details.
"""

from __future__ import annotations

_DATA_VERSION = "2025-01"

_ACID = {
    "A": {"name": "Atomicity", "description": "Transaction is all-or-nothing. If any part fails, entire transaction rolls back.", "example": "Bank transfer: debit + credit both succeed or neither does"},
    "C": {"name": "Consistency", "description": "Transaction takes database from one valid state to another. All constraints satisfied.", "example": "Foreign keys, unique constraints, check constraints all hold after commit"},
    "I": {"name": "Isolation", "description": "Concurrent transactions don't interfere with each other.", "levels": ["READ UNCOMMITTED", "READ COMMITTED", "REPEATABLE READ", "SERIALIZABLE"]},
    "D": {"name": "Durability", "description": "Once committed, transaction survives crashes (written to non-volatile storage).", "mechanism": "WAL (Write-Ahead Log), fsync"},
}

_CAP = {
    "C": {"name": "Consistency", "description": "Every read receives the most recent write or an error", "note": "Different from ACID consistency — this means linearizability"},
    "A": {"name": "Availability", "description": "Every request receives a response (not necessarily most recent data)"},
    "P": {"name": "Partition tolerance", "description": "System continues operating despite network partitions between nodes"},
    "theorem": "In a distributed system, you can only guarantee 2 of 3 (C, A, P). Since network partitions are inevitable, the real choice is CP (consistent but may be unavailable) vs AP (available but may be stale).",
    "CP_examples": ["MongoDB (default)", "HBase", "Redis Cluster", "Zookeeper"],
    "AP_examples": ["Cassandra", "DynamoDB", "CouchDB", "Riak"],
}

_NORMAL_FORMS = {
    "1NF": {"name": "First Normal Form", "rule": "No repeating groups. Each column has atomic (indivisible) values.", "violation": "Column storing comma-separated tags: 'python,rust,go'", "fix": "Separate table for tags with foreign key"},
    "2NF": {"name": "Second Normal Form", "requires": "1NF", "rule": "No partial dependency — every non-key column depends on the ENTIRE primary key.", "applies_to": "composite keys only", "violation": "Table(student_id, course_id, student_name) — student_name depends only on student_id"},
    "3NF": {"name": "Third Normal Form", "requires": "2NF", "rule": "No transitive dependency — non-key columns don't depend on other non-key columns.", "violation": "Table(id, zip, city) — city depends on zip, not directly on id"},
    "BCNF": {"name": "Boyce-Codd Normal Form", "requires": "3NF", "rule": "Every determinant is a candidate key.", "stricter_than": "3NF — handles edge cases where 3NF allows some anomalies"},
}

_INDEX_TYPES = {
    "B-tree": {"description": "Balanced tree, default for most databases", "good_for": "equality, range, sorting, prefix", "bad_for": "full-text search", "used_by": ["PostgreSQL", "MySQL InnoDB", "SQLite"]},
    "hash": {"description": "Hash table index", "good_for": "exact equality lookups", "bad_for": "range queries, sorting", "used_by": ["PostgreSQL", "Redis"]},
    "GiST": {"description": "Generalized Search Tree", "good_for": "geometric data, full-text, ranges", "used_by": ["PostgreSQL"]},
    "GIN": {"description": "Generalized Inverted Index", "good_for": "full-text search, arrays, JSONB", "used_by": ["PostgreSQL"]},
    "BRIN": {"description": "Block Range INdex", "good_for": "large naturally-ordered tables (time series)", "space": "very small", "used_by": ["PostgreSQL"]},
    "bitmap": {"description": "Bitmap index", "good_for": "low-cardinality columns (gender, status)", "bad_for": "high-cardinality, frequent updates", "used_by": ["Oracle"]},
    "clustered": {"description": "Data physically sorted by index key (one per table)", "good_for": "range scans on the key", "used_by": ["SQL Server (clustered)", "MySQL InnoDB (primary key is clustered)"]},
    "covering": {"description": "Index contains all columns needed by query (index-only scan)", "good_for": "avoiding table lookups", "syntax": "CREATE INDEX ... INCLUDE (col1, col2)"},
}

_SQL_VS_NOSQL = {
    "SQL": {"model": "relational (tables, rows, joins)", "schema": "fixed, enforced", "query": "SQL", "scaling": "vertical (primary)", "acid": True, "examples": ["PostgreSQL", "MySQL", "SQLite", "Oracle", "SQL Server"], "use_when": "complex queries, joins, transactions, structured data"},
    "NoSQL": {"models": ["document (MongoDB)", "key-value (Redis, DynamoDB)", "wide-column (Cassandra)", "graph (Neo4j)"], "schema": "flexible/schemaless", "query": "varies (API, query language)", "scaling": "horizontal (primary)", "acid": "varies (some support)", "use_when": "high scale, flexible schema, specific access patterns"},
}


def acid_property(letter: str) -> dict:
    """Get ACID property by letter."""
    key = str(letter).upper().strip()[0]
    entry = _ACID.get(key)
    if not entry:
        return {"error": f"Unknown: {letter}", "valid": ["A", "C", "I", "D"]}
    return {"letter": key, **entry}


def all_acid() -> dict:
    """Get all ACID properties."""
    return {k: v["name"] + ": " + v["description"] for k, v in _ACID.items()}


def cap_theorem() -> dict:
    """Explain the CAP theorem."""
    return _CAP


def normal_form(nf: str) -> dict:
    """Get normalization form details (1NF, 2NF, 3NF, BCNF)."""
    key = str(nf).upper().strip()
    if not key.endswith("NF"):
        key += "NF"
    entry = _NORMAL_FORMS.get(key)
    if not entry:
        return {"error": f"Unknown: {nf}", "valid": list(_NORMAL_FORMS.keys())}
    return {"form": key, **entry}


def index_type(name: str) -> dict:
    """Get database index type details."""
    key = str(name).strip()
    for k, v in _INDEX_TYPES.items():
        if key.lower() in k.lower():
            return {"type": k, **v}
    return {"error": f"Unknown: {name}", "valid": list(_INDEX_TYPES.keys())}


def sql_vs_nosql() -> dict:
    """Compare SQL and NoSQL databases."""
    return _SQL_VS_NOSQL


def list_index_types() -> list[str]:
    """List all known index types."""
    return list(_INDEX_TYPES.keys())


DATABASE_FUNCTIONS = {
    "acid_property": acid_property,
    "all_acid": all_acid,
    "cap_theorem": cap_theorem,
    "normal_form": normal_form,
    "index_type": index_type,
    "sql_vs_nosql": sql_vs_nosql,
    "list_index_types": list_index_types,
}

DATABASE_NL_PATTERNS = [
    (r'(?:what is|explain)\s+(?:the\s+)?([ACID])\s+(?:in|of|from|property)\s+ACID', 'acid_property("{0}")'),
    (r'(?:what is|explain)\s+ACID', 'all_acid()'),
    (r'(?:what is|explain)\s+(?:the\s+)?CAP\s+theorem', 'cap_theorem()'),
    (r'(?:what is|explain)\s+(1NF|2NF|3NF|BCNF|first|second|third)\s+(?:normal\s+form|NF)', 'normal_form("{0}")'),
    (r'(?:what is|explain)\s+(?:a\s+)?(B.tree|hash|GiST|GIN|BRIN|bitmap|clustered|covering)\s+index', 'index_type("{0}")'),
    (r'(?:compare|difference|vs)\s+SQL\s+(?:and|vs)\s+NoSQL', 'sql_vs_nosql()'),
]
