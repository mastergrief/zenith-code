"""
CALM Encoding reference knowledge backend — character encodings, escaping, formats.

Models confuse UTF-8 vs UTF-16, hallucinate byte sequences, mix up encodings.
"""

from __future__ import annotations

_DATA_VERSION = "2025-01"

_ENCODINGS = {
    "ASCII": {"bits": 7, "bytes_per_char": 1, "range": "0-127", "chars": 128, "description": "Original English-only encoding. Subset of UTF-8.", "use": "simple English text, protocol headers"},
    "UTF-8": {"bits": "variable (8-32)", "bytes_per_char": "1-4", "description": "Variable-width, backward compatible with ASCII. Dominant encoding on the web (~98%).", "bom": "optional (EF BB BF, discouraged)", "use": "everything — web, APIs, databases, files"},
    "UTF-16": {"bits": "variable (16 or 32)", "bytes_per_char": "2 or 4 (surrogate pairs)", "description": "Used by Windows internals, Java, JavaScript strings.", "bom": "required (FF FE or FE FF)", "endian": "LE or BE", "gotcha": "Chars outside BMP require surrogate pairs"},
    "UTF-32": {"bits": 32, "bytes_per_char": 4, "description": "Fixed-width, every char is 4 bytes. Wasteful but simple indexing.", "use": "internal processing only — too large for storage/network"},
    "Latin-1": {"bits": 8, "bytes_per_char": 1, "range": "0-255", "chars": 256, "alias": "ISO 8859-1", "description": "Extends ASCII with Western European chars (ñ, ü, é).", "gotcha": "NOT the same as Windows-1252 (which adds smart quotes, em dash)"},
    "Windows-1252": {"bits": 8, "bytes_per_char": 1, "range": "0-255", "alias": "CP-1252", "description": "Microsoft's extension of Latin-1. Adds smart quotes, em dash, euro sign.", "gotcha": "Often mislabeled as Latin-1. Bytes 0x80-0x9F differ from ISO 8859-1."},
    "Shift-JIS": {"bits": "variable (8-16)", "bytes_per_char": "1-2", "description": "Japanese encoding. Legacy — use UTF-8 for new systems.", "gotcha": "Backslash (0x5C) conflicts with Yen sign"},
    "EUC-JP": {"bits": "variable (8-24)", "bytes_per_char": "1-3", "description": "Japanese encoding used on Unix/Linux. Legacy."},
    "GB2312": {"bits": "variable", "bytes_per_char": "1-2", "description": "Simplified Chinese. Superseded by GBK/GB18030."},
    "EUC-KR": {"bits": "variable", "bytes_per_char": "1-2", "description": "Korean encoding. Legacy."},
}

_ESCAPE_SEQUENCES = {
    "URL": {"method": "percent-encoding", "space": "%20 or +", "example": "hello world → hello%20world", "standard": "RFC 3986"},
    "HTML": {"method": "character references", "examples": {"<": "&lt;", ">": "&gt;", "&": "&amp;", '"': "&quot;", "'": "&#39;"}, "standard": "HTML5"},
    "JSON": {"method": "backslash escaping", "examples": {'"': '\\"', "\\": "\\\\", "\n": "\\n", "\t": "\\t", "\r": "\\r"}, "unicode": "\\uXXXX"},
    "regex": {"method": "backslash escaping", "metacharacters": ". * + ? ^ $ { } [ ] ( ) | \\", "literal_dot": "\\."},
    "SQL": {"method": "single quote doubling or parameterized queries", "examples": {"'": "''", "\\": "\\\\"}, "recommendation": "ALWAYS use parameterized queries, never string escaping"},
    "shell": {"method": "single quotes, double quotes, or backslash", "single_quotes": "preserve everything literally (except ' itself)", "double_quotes": "allow $var, `cmd`, \\", "example": "echo 'hello world'"},
    "C/Python": {"method": "backslash sequences", "examples": {"\\n": "newline", "\\t": "tab", "\\r": "carriage return", "\\0": "null", "\\\\": "backslash", "\\\"": "double quote"}},
}

_BOM = {
    "UTF-8": {"bytes": "EF BB BF", "required": False, "note": "Discouraged. Can cause issues with Unix tools."},
    "UTF-16 LE": {"bytes": "FF FE", "required": True},
    "UTF-16 BE": {"bytes": "FE FF", "required": True},
    "UTF-32 LE": {"bytes": "FF FE 00 00", "required": True},
    "UTF-32 BE": {"bytes": "00 00 FE FF", "required": True},
}

_LINE_ENDINGS = {
    "LF": {"char": "\\n", "hex": "0x0A", "os": "Unix/Linux/macOS", "standard": "POSIX"},
    "CRLF": {"char": "\\r\\n", "hex": "0x0D 0x0A", "os": "Windows", "note": "HTTP protocol requires CRLF"},
    "CR": {"char": "\\r", "hex": "0x0D", "os": "Classic Mac OS (pre-OS X)", "note": "Essentially obsolete"},
}


def encoding_info(name: str) -> dict:
    """Get details about a character encoding."""
    key = str(name).strip()
    for k, v in _ENCODINGS.items():
        if key.lower().replace("-", "").replace("_", "") == k.lower().replace("-", "").replace("_", ""):
            return {"encoding": k, **v}
        if key.lower() == v.get("alias", "").lower().replace(" ", ""):
            return {"encoding": k, **v}
    return {"error": f"Unknown: {name}", "valid": list(_ENCODINGS.keys())}


def escape_rules(context: str) -> dict:
    """Get escaping rules for a context (URL, HTML, JSON, regex, SQL, shell, C)."""
    key = str(context).strip()
    for k, v in _ESCAPE_SEQUENCES.items():
        if key.lower() in k.lower() or k.lower() in key.lower():
            return {"context": k, **v}
    return {"error": f"Unknown: {context}", "valid": list(_ESCAPE_SEQUENCES.keys())}


def bom_info(encoding: str) -> dict:
    """Get BOM (Byte Order Mark) for an encoding."""
    key = str(encoding).upper().strip()
    for k, v in _BOM.items():
        if key in k:
            return {"encoding": k, **v}
    return {"error": f"Unknown: {encoding}", "valid": list(_BOM.keys())}


def line_ending(name: str) -> dict:
    """Get line ending details (LF, CRLF, CR)."""
    key = str(name).upper().strip()
    entry = _LINE_ENDINGS.get(key)
    if not entry:
        return {"error": f"Unknown: {name}", "valid": list(_LINE_ENDINGS.keys())}
    return {"ending": key, **entry}


def utf8_bytes(char: str) -> list[str]:
    """Get UTF-8 byte sequence for a character as hex strings."""
    return [f"0x{b:02X}" for b in str(char)[0].encode('utf-8')]


def utf8_vs_utf16() -> dict:
    """Compare UTF-8 and UTF-16."""
    return {
        "UTF-8": {"ASCII_compatible": True, "web_share": "~98%", "min_bytes": 1, "max_bytes": 4, "endian": "no (byte-oriented)"},
        "UTF-16": {"ASCII_compatible": False, "web_share": "~0.01%", "min_bytes": 2, "max_bytes": 4, "endian": "yes (LE/BE)"},
        "recommendation": "Use UTF-8 for everything unless a platform requires UTF-16 (Windows API, Java, JavaScript internals).",
    }


def crlf_vs_lf() -> dict:
    """Compare line endings."""
    return _LINE_ENDINGS


ENCODING_REF_FUNCTIONS = {
    "encoding_info": encoding_info,
    "escape_rules": escape_rules,
    "bom_info": bom_info,
    "line_ending": line_ending,
    "utf8_bytes": utf8_bytes,
    "utf8_vs_utf16": utf8_vs_utf16,
    "crlf_vs_lf": crlf_vs_lf,
}

ENCODING_REF_NL_PATTERNS = [
    (r'(?:what is|explain)\s+(UTF-?8|UTF-?16|UTF-?32|ASCII|Latin-?1|Shift-?JIS|Windows-?1252|EUC)', 'encoding_info("{0}")'),
    (r'(?:how to|how do you)\s+escape\s+(?:in|for)\s+(URL|HTML|JSON|regex|SQL|shell|C|Python)', 'escape_rules("{0}")'),
    (r'(?:compare|difference|vs)\s+UTF-?8\s+(?:and|vs)\s+UTF-?16', 'utf8_vs_utf16()'),
    (r'(?:compare|difference|vs)\s+(?:CRLF|CR)\s+(?:and|vs)\s+LF', 'crlf_vs_lf()'),
    (r'(?:what is|explain)\s+(?:the\s+)?BOM\s+(?:for|of|in)\s+(\w+)', 'bom_info("{0}")'),
    (r'(?:what is|explain)\s+(?:line ending|newline|EOL)\s+(LF|CRLF|CR)', 'line_ending("{0}")'),
]
