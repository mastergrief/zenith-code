"""
CALM Phonetics backend — Soundex, Metaphone for fuzzy name matching.

Models can't compute phonetic codes reliably. Pure algorithmic.
"""

from __future__ import annotations


def soundex(name: str) -> str:
    """American Soundex code for a name (4-char, e.g. Robert → R163)."""
    s = ''.join(c for c in str(name).upper() if c.isalpha())
    if not s:
        return "0000"
    # Soundex coding table
    coding = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6',
    }
    result = s[0]
    prev = coding.get(s[0], '0')
    for c in s[1:]:
        code = coding.get(c, '0')
        if code != '0' and code != prev:
            result += code
        # H and W don't reset prev (they're transparent)
        if c not in ('H', 'W'):
            prev = code
        if len(result) == 4:
            break
    return (result + '0000')[:4]


def metaphone(name: str) -> str:
    """Original Metaphone phonetic code (variable length)."""
    s = ''.join(c for c in str(name).upper() if c.isalpha())
    if not s:
        return ""

    # Drop initial silent letters
    if len(s) >= 2 and s[:2] in ('AE', 'GN', 'KN', 'PN', 'WR'):
        s = s[1:]

    result = []
    i = 0
    while i < len(s):
        c = s[i]
        # Skip duplicate adjacent letters (except C)
        if i > 0 and c == s[i - 1] and c != 'C':
            i += 1
            continue

        if c in 'AEIOU':
            if i == 0:
                result.append(c)
        elif c == 'B':
            if i == 0 or s[i - 1] != 'M':
                result.append('B')
        elif c == 'C':
            if i + 1 < len(s) and s[i + 1] in 'EIY':
                result.append('S')
            else:
                result.append('K')
        elif c == 'D':
            if i + 1 < len(s) and s[i + 1] in 'GEI':
                result.append('J')
            else:
                result.append('T')
        elif c == 'F':
            result.append('F')
        elif c == 'G':
            if i + 1 < len(s) and s[i + 1] in 'EIY':
                result.append('J')
            elif i > 0 and s[i - 1] in 'AEIOU' and (i + 1 >= len(s) or s[i + 1] not in 'AEIOU'):
                pass  # silent G
            else:
                result.append('K')
        elif c == 'H':
            if i + 1 < len(s) and s[i + 1] in 'AEIOU' and (i == 0 or s[i - 1] not in 'AEIOU'):
                result.append('H')
        elif c == 'J':
            result.append('J')
        elif c == 'K':
            if i == 0 or s[i - 1] != 'C':
                result.append('K')
        elif c == 'L':
            result.append('L')
        elif c == 'M':
            result.append('M')
        elif c == 'N':
            result.append('N')
        elif c == 'P':
            if i + 1 < len(s) and s[i + 1] == 'H':
                result.append('F')
                i += 1
            else:
                result.append('P')
        elif c == 'Q':
            result.append('K')
        elif c == 'R':
            result.append('R')
        elif c == 'S':
            if i + 1 < len(s) and s[i + 1] == 'H':
                result.append('X')
                i += 1
            elif i + 2 < len(s) and s[i:i + 3] == 'SIO':
                result.append('X')
                i += 2
            elif i + 2 < len(s) and s[i:i + 3] == 'SIA':
                result.append('X')
                i += 2
            else:
                result.append('S')
        elif c == 'T':
            if i + 1 < len(s) and s[i + 1] == 'H':
                result.append('0')  # theta
                i += 1
            elif i + 2 < len(s) and s[i:i + 3] in ('TIA', 'TIO'):
                result.append('X')
                i += 2
            else:
                result.append('T')
        elif c == 'V':
            result.append('F')
        elif c == 'W':
            if i + 1 < len(s) and s[i + 1] in 'AEIOU':
                result.append('W')
        elif c == 'X':
            result.append('K')
            result.append('S')
        elif c == 'Y':
            if i + 1 < len(s) and s[i + 1] in 'AEIOU':
                result.append('Y')
        elif c == 'Z':
            result.append('S')

        i += 1

    return ''.join(result)


def soundex_match(name1: str, name2: str) -> bool:
    """Whether two names have the same Soundex code."""
    return soundex(name1) == soundex(name2)


def metaphone_match(name1: str, name2: str) -> bool:
    """Whether two names have the same Metaphone code."""
    return metaphone(name1) == metaphone(name2)


def nysiis(name: str) -> str:
    """New York State Identification and Intelligence System phonetic code."""
    s = ''.join(c for c in str(name).upper() if c.isalpha())
    if not s:
        return ""

    # Translate first characters
    for old, new in [('MAC', 'MCC'), ('KN', 'NN'), ('K', 'C'), ('PH', 'FF'),
                     ('PF', 'FF'), ('SCH', 'SSS')]:
        if s.startswith(old):
            s = new + s[len(old):]
            break

    # Translate last characters
    for old, new in [('EE', 'Y'), ('IE', 'Y'), ('DT', 'D'), ('RT', 'D'),
                     ('RD', 'D'), ('NT', 'D'), ('ND', 'D')]:
        if s.endswith(old):
            s = s[:-len(old)] + new
            break

    first = s[0]
    result = [first]
    i = 1
    while i < len(s):
        c = s[i]
        if c in 'AEIOU':
            result.append('A')
        elif c == 'Q':
            result.append('G')
        elif c == 'Z':
            result.append('S')
        elif c == 'M':
            result.append('N')
        elif c == 'K':
            if i + 1 < len(s) and s[i + 1] == 'N':
                result.append('N')
            else:
                result.append('C')
        elif c == 'H':
            prev = s[i - 1] if i > 0 else ''
            nxt = s[i + 1] if i + 1 < len(s) else ''
            if prev not in 'AEIOU' or nxt not in 'AEIOU':
                result.append(result[-1] if result else c)
            else:
                result.append('H')
        elif c == 'W':
            prev = s[i - 1] if i > 0 else ''
            if prev in 'AEIOU':
                result.append(result[-1] if result else c)
            else:
                result.append('W')
        else:
            result.append(c)
        i += 1

    # Remove trailing S
    code = ''.join(result)
    if code.endswith('S') and len(code) > 1:
        code = code[:-1]
    # Replace trailing AY with Y
    if code.endswith('AY'):
        code = code[:-2] + 'Y'
    # Remove trailing A
    if code.endswith('A') and len(code) > 1:
        code = code[:-1]
    # Remove consecutive duplicates
    final = [code[0]]
    for c in code[1:]:
        if c != final[-1]:
            final.append(c)
    return ''.join(final)[:6]


def levenshtein(s1: str, s2: str) -> int:
    """Levenshtein edit distance between two strings."""
    a, b = str(s1), str(s2)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (0 if ca == cb else 1)))
        prev = curr
    return prev[-1]


PHONETICS_FUNCTIONS = {
    "soundex": soundex,
    "metaphone": metaphone,
    "soundex_match": soundex_match,
    "metaphone_match": metaphone_match,
    "nysiis": nysiis,
    "levenshtein": levenshtein,
}

PHONETICS_NL_PATTERNS = [
    (r'soundex\s+(?:of|for|code)\s+["\']?(\w+)["\']?', 'soundex("{0}")'),
    (r'metaphone\s+(?:of|for|code)\s+["\']?(\w+)["\']?', 'metaphone("{0}")'),
    (r'(?:do|does)\s+["\']?(\w+)["\']?\s+(?:and)\s+["\']?(\w+)["\']?\s+(?:sound alike|match|rhyme)', 'soundex_match("{0}", "{1}")'),
    (r'(?:edit|levenshtein)\s+distance\s+(?:between|of|from)\s+["\']?(\w+)["\']?\s+(?:and|to)\s+["\']?(\w+)["\']?', 'levenshtein("{0}", "{1}")'),
]
