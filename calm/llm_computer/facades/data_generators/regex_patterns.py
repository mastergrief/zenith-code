"""RegexPatternsGenerator — validation / extraction via regex.

Every example produces a small Python function that uses `re.match` or
`re.findall` to validate or extract structured data. Covers the
everyday patterns Gemma generates (often with subtle bugs around
anchors, escaping, character classes).

All solutions are sandbox-safe (only `import re` which the sandbox
allows) and each has deterministic test cases.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import List, Tuple

from calm.llm_computer.facades.data_generators import register_generator
from calm.llm_computer.facades.data_generators.base import (
    DomainDataGenerator,
    VerifiedExample,
)


@dataclass
class RegexSpec:
    name: str
    fn_name: str
    pattern: str
    usage: str              # one of "match", "fullmatch", "findall", "search"
    problem: str
    test_cases: List[Tuple]
    algorithm: str
    edge_cases: List[str]


def _specs() -> List[RegexSpec]:
    out: List[RegexSpec] = []

    out.append(RegexSpec(
        name="email_basic",
        fn_name="is_email",
        pattern=r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$",
        usage="fullmatch",
        problem="Write a Python function `is_email(s)` that returns True if s looks like a simple email address. Must anchor with ^ and $, allow dot-separated local/domain parts, require a TLD of at least 2 letters.",
        test_cases=[
            ("alice@example.com", True),
            ("alice+tag@example.co.uk", True),
            ("invalid", False),
            ("@example.com", False),
            ("alice@", False),
            ("alice@example", False),
            ("alice@example.c", False),
            ("alice@@example.com", False),
            ("", False),
        ],
        algorithm="anchored regex, no nested quantifiers (DoS-safe)",
        edge_cases=["missing @", "missing TLD", "TLD too short", "empty", "double @"],
    ))

    out.append(RegexSpec(
        name="ipv4",
        fn_name="is_ipv4",
        pattern=r"^(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)$",
        usage="fullmatch",
        problem="Write a Python function `is_ipv4(s)` that returns True if s is a valid IPv4 address (four 0-255 octets separated by dots, no leading zeros beyond '0' itself).",
        test_cases=[
            ("0.0.0.0", True),
            ("255.255.255.255", True),
            ("192.168.1.1", True),
            ("1.2.3.4", True),
            ("256.1.1.1", False),
            ("1.2.3", False),
            ("1.2.3.4.5", False),
            ("", False),
            ("abc.def.ghi.jkl", False),
        ],
        algorithm="alternation per octet with 0-255 range, 4-group structure",
        edge_cases=["0.0.0.0 and 255.255.255.255 boundary", "fewer/more octets", "non-digit chars"],
    ))

    out.append(RegexSpec(
        name="uuid_v4",
        fn_name="is_uuid_v4",
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
        usage="fullmatch",
        problem="Write a Python function `is_uuid_v4(s)` that returns True if s is a lowercase-formatted UUID v4 (the v4 variant has a fixed '4' in position 14 and '8', '9', 'a', or 'b' in position 19).",
        test_cases=[
            ("f47ac10b-58cc-4372-a567-0e02b2c3d479", True),
            ("f47ac10b-58cc-1372-a567-0e02b2c3d479", False),    # not v4
            ("f47ac10b-58cc-4372-c567-0e02b2c3d479", False),    # wrong variant
            ("not-a-uuid", False),
            ("F47AC10B-58CC-4372-A567-0E02B2C3D479", False),    # upper case
            ("", False),
        ],
        algorithm="v4-specific position constraints in regex",
        edge_cases=["case sensitivity (lower only)", "variant byte", "version byte"],
    ))

    out.append(RegexSpec(
        name="hex_color",
        fn_name="is_hex_color",
        pattern=r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$",
        usage="fullmatch",
        problem="Write a Python function `is_hex_color(s)` that returns True if s is a CSS hex color: '#' followed by exactly 3 or 6 hex digits.",
        test_cases=[
            ("#fff", True),
            ("#FFF", True),
            ("#ffffff", True),
            ("#123abc", True),
            ("fff", False),           # missing #
            ("#ffff", False),         # 4 digits
            ("#gggggg", False),       # non-hex
            ("#", False),
            ("", False),
        ],
        algorithm="alternation between 3-char and 6-char hex body",
        edge_cases=["no '#'", "4 or 5 digits invalid", "case insensitive", "non-hex chars"],
    ))

    out.append(RegexSpec(
        name="iso_date",
        fn_name="is_iso_date",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        usage="fullmatch",
        problem="Write a Python function `is_iso_date(s)` that returns True if s is a YYYY-MM-DD formatted string (syntactic check only — doesn't validate that the date is real).",
        test_cases=[
            ("2024-01-15", True),
            ("1999-12-31", True),
            ("0001-01-01", True),
            ("2024-1-15", False),     # missing zero pad
            ("2024-01-15T00:00:00", False),
            ("", False),
            ("Jan 15, 2024", False),
        ],
        algorithm="anchored regex with fixed-width digit groups",
        edge_cases=["zero-padding required", "time part rejected", "empty"],
    ))

    out.append(RegexSpec(
        name="extract_numbers",
        fn_name="extract_numbers",
        pattern=r"-?\d+(?:\.\d+)?",
        usage="findall",
        problem="Write a Python function `extract_numbers(text)` that returns a list of all integer and decimal numbers found in text (may have a leading minus). Return them in order of appearance.",
        test_cases=[
            ("", []),
            ("no numbers here", []),
            ("I have 5 apples", ["5"]),
            ("The temperature is -40.5 degrees", ["-40.5"]),
            ("Coords: 3.14, -2.7, 100", ["3.14", "-2.7", "100"]),
            ("Version 1.2.3", ["1.2", "3"]),      # dot between two numbers is consumed once
        ],
        algorithm="findall with optional sign and decimal part",
        edge_cases=["empty", "no matches", "negatives", "decimals", "two numbers separated by dot"],
    ))

    out.append(RegexSpec(
        name="extract_urls",
        fn_name="extract_urls",
        pattern=r"https?://[^\s<>\"']+",
        usage="findall",
        problem="Write a Python function `extract_urls(text)` that returns a list of http(s) URLs from the text. URLs are whitespace- and quote-terminated.",
        test_cases=[
            ("", []),
            ("just text", []),
            ("visit https://example.com", ["https://example.com"]),
            ("http://a.com and https://b.org", ["http://a.com", "https://b.org"]),
            ("(see https://c.com)", ["https://c.com)"]),   # paren kept unless explicitly excluded
            ("wrap it in \"https://d.com\" quotes", ["https://d.com"]),
        ],
        algorithm="findall with scheme prefix + non-whitespace/non-quote char class",
        edge_cases=["empty", "multiple URLs", "quote termination", "paren boundary"],
    ))

    out.append(RegexSpec(
        name="is_phone_e164",
        fn_name="is_phone_e164",
        pattern=r"^\+[1-9]\d{1,14}$",
        usage="fullmatch",
        problem="Write a Python function `is_phone_e164(s)` that returns True if s is a valid E.164 phone number: leading '+', country code 1-9 start, 1-15 digits total after the +.",
        test_cases=[
            ("+14155551212", True),
            ("+442071838750", True),
            ("+1", False),           # too short (min 2 digits after +)
            ("+0123", False),        # leading 0 after +
            ("14155551212", False),  # missing +
            ("", False),
        ],
        algorithm="anchored + with digit-range + length bound",
        edge_cases=["no +", "too short/long", "leading 0"],
    ))

    out.append(RegexSpec(
        name="is_semver",
        fn_name="is_semver",
        pattern=r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$",
        usage="fullmatch",
        problem="Write a Python function `is_semver(s)` that returns True if s is a valid semver version (major.minor.patch with optional pre-release and build metadata). Must not have leading zeros in numeric parts.",
        test_cases=[
            ("1.0.0", True),
            ("1.2.3", True),
            ("0.0.0", True),
            ("1.0.0-alpha", True),
            ("1.0.0-alpha.1+build.123", True),
            ("1.2", False),           # not 3 parts
            ("01.0.0", False),        # leading zero
            ("1.0.0.0", False),       # 4 parts
            ("", False),
        ],
        algorithm="three no-leading-zero numeric groups with optional pre-release/build",
        edge_cases=["leading zeros rejected", "pre-release and build optional", "patch required"],
    ))

    out.append(RegexSpec(
        name="strip_comments",
        fn_name="strip_py_comments",
        pattern=r"(?m)^\s*#.*$|\s+#.*$",
        usage="sub",   # special-case: substitution
        problem="Write a Python function `strip_py_comments(code)` that removes whole-line and trailing Python comments, preserving indentation of remaining lines. Returns the stripped source.",
        test_cases=[
            ("", ""),
            ("x = 1", "x = 1"),
            ("# full line comment", ""),
            ("x = 1  # trailing", "x = 1"),
            ("  # indented comment\n", ""),
            ("a = 1\n# comment\nb = 2", "a = 1\n\nb = 2"),
            ("s = 'hash # in string'  # comment", "s = 'hash # in string'"),
        ],
        algorithm="regex with multi-line mode, handles whole-line and trailing",
        edge_cases=["full-line vs trailing", "indented comments", "hash inside quoted string (this simple regex doesn't handle — acceptable)"],
    ))

    return out


# Note: we use `{pattern!r}` (Python repr) — NOT `r{pattern!r}`.
# The raw-string r-prefix combined with repr's backslash escaping would
# produce `\\.` that means LITERAL backslash-period in the regex, not
# an escaped period. repr() alone produces a correctly-quoted string.
_SOLUTION_TEMPLATES = {
    "match": (
        "def {fn}(s):\n"
        "    import re\n"
        "    return re.match({pattern!r}, s) is not None\n"
    ),
    "fullmatch": (
        "def {fn}(s):\n"
        "    import re\n"
        "    return re.fullmatch({pattern!r}, s) is not None\n"
    ),
    "findall": (
        "def {fn}(text):\n"
        "    import re\n"
        "    return re.findall({pattern!r}, text)\n"
    ),
    "search": (
        "def {fn}(text):\n"
        "    import re\n"
        "    return re.search({pattern!r}, text) is not None\n"
    ),
    "sub": (
        "def {fn}(code):\n"
        "    import re\n"
        "    return re.sub({pattern!r}, '', code)\n"
    ),
}


def _build_example(spec: RegexSpec) -> VerifiedExample:
    tmpl = _SOLUTION_TEMPLATES[spec.usage]
    solution = tmpl.format(fn=spec.fn_name, pattern=spec.pattern)
    return VerifiedExample(
        problem=spec.problem,
        signature=f"def {spec.fn_name}(" + ("s" if spec.usage in ("match", "fullmatch") else
                                            ("code" if spec.usage == "sub" else "text")) + "):",
        solution=solution,
        test_cases=list(spec.test_cases),
        reasoning="",
        algorithm=spec.algorithm,
        complexity="O(|input|) for anchored/linear regex",
        edge_cases=list(spec.edge_cases),
        category=f"regex_{spec.name}",
        generator_name="regex",
    )


class RegexPatternsGenerator(DomainDataGenerator):
    """Regex-validation / extraction problems. 10 canonical patterns
    (email, IPv4, UUID, hex color, ISO date, number extract, URL extract,
    phone E.164, semver, comment strip)."""

    name = "regex"

    def __init__(self, rng=None):
        super().__init__(rng)
        self._specs = _specs()

    def generate_raw(self, n: int) -> List[VerifiedExample]:
        self.rng.shuffle(self._specs)
        return [_build_example(s) for s in self._specs[:n]]


register_generator("regex", RegexPatternsGenerator)
