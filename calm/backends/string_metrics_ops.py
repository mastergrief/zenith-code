"""
CALM String metrics backend — similarity, distance, matching algorithms.

Models approximate string distances. Pure computation.
"""

from __future__ import annotations


def hamming_distance(s1: str, s2: str) -> int:
    """Hamming distance: count of positions where chars differ. Equal-length strings only."""
    a, b = str(s1), str(s2)
    if len(a) != len(b):
        return -1
    return sum(c1 != c2 for c1, c2 in zip(a, b))


def jaccard_similarity(s1: str, s2: str) -> float:
    """Jaccard similarity of word sets: |intersection| / |union|."""
    set1 = set(str(s1).lower().split())
    set2 = set(str(s2).lower().split())
    if not set1 and not set2:
        return 1.0
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return round(inter / union, 4) if union else 0.0


def cosine_similarity_words(s1: str, s2: str) -> float:
    """Cosine similarity of word frequency vectors."""
    from collections import Counter
    import math
    w1 = Counter(str(s1).lower().split())
    w2 = Counter(str(s2).lower().split())
    all_words = set(w1.keys()) | set(w2.keys())
    dot = sum(w1.get(w, 0) * w2.get(w, 0) for w in all_words)
    mag1 = math.sqrt(sum(v ** 2 for v in w1.values()))
    mag2 = math.sqrt(sum(v ** 2 for v in w2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return round(dot / (mag1 * mag2), 4)


def longest_common_substring(s1: str, s2: str) -> str:
    """Longest common substring between two strings."""
    a, b = str(s1), str(s2)
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return ""
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0
    end_idx = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    end_idx = i
    return a[end_idx - max_len:end_idx]


def longest_common_subsequence(s1: str, s2: str) -> int:
    """Length of longest common subsequence."""
    a, b = str(s1), str(s2)
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def damerau_levenshtein(s1: str, s2: str) -> int:
    """Damerau-Levenshtein distance (allows transpositions)."""
    a, b = str(s1), str(s2)
    la, lb = len(a), len(b)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + cost)
    return d[la][lb]


def jaro_similarity(s1: str, s2: str) -> float:
    """Jaro similarity (0 to 1)."""
    a, b = str(s1), str(s2)
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    match_dist = max(la, lb) // 2 - 1
    a_matches = [False] * la
    b_matches = [False] * lb
    matches = 0
    transpositions = 0
    for i in range(la):
        start = max(0, i - match_dist)
        end = min(i + match_dist + 1, lb)
        for j in range(start, end):
            if b_matches[j] or a[i] != b[j]:
                continue
            a_matches[i] = True
            b_matches[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    k = 0
    for i in range(la):
        if not a_matches[i]:
            continue
        while not b_matches[k]:
            k += 1
        if a[i] != b[k]:
            transpositions += 1
        k += 1
    return round((matches / la + matches / lb + (matches - transpositions / 2) / matches) / 3, 4)


def jaro_winkler(s1: str, s2: str, p: float = 0.1) -> float:
    """Jaro-Winkler similarity (boosts for common prefix)."""
    jaro = jaro_similarity(s1, s2)
    a, b = str(s1), str(s2)
    prefix = 0
    for i in range(min(4, len(a), len(b))):
        if a[i] == b[i]:
            prefix += 1
        else:
            break
    return round(jaro + prefix * float(p) * (1 - jaro), 4)


def sorensen_dice(s1: str, s2: str) -> float:
    """Sørensen-Dice coefficient on character bigrams."""
    a, b = str(s1).lower(), str(s2).lower()
    if len(a) < 2 or len(b) < 2:
        return 1.0 if a == b else 0.0
    bg1 = set(a[i:i + 2] for i in range(len(a) - 1))
    bg2 = set(b[i:i + 2] for i in range(len(b) - 1))
    inter = len(bg1 & bg2)
    return round(2 * inter / (len(bg1) + len(bg2)), 4) if (len(bg1) + len(bg2)) else 0.0


def normalized_levenshtein(s1: str, s2: str) -> float:
    """Normalized Levenshtein distance (0 = identical, 1 = completely different)."""
    a, b = str(s1), str(s2)
    if not a and not b:
        return 0.0
    # Inline levenshtein
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (0 if ca == cb else 1)))
        prev = curr
    return round(prev[-1] / max(len(a), len(b)), 4)


def is_anagram(s1: str, s2: str) -> bool:
    """Whether two strings are anagrams (same letters, different order)."""
    return sorted(str(s1).lower().replace(' ', '')) == sorted(str(s2).lower().replace(' ', ''))


def is_palindrome(s: str) -> bool:
    """Whether a string is a palindrome (ignoring case and spaces)."""
    cleaned = ''.join(c.lower() for c in str(s) if c.isalnum())
    return cleaned == cleaned[::-1]


def common_prefix(strings: list) -> str:
    """Longest common prefix of a list of strings."""
    if not strings:
        return ""
    strs = [str(s) for s in strings]
    prefix = strs[0]
    for s in strs[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix


STRING_METRICS_FUNCTIONS = {
    "hamming_distance": hamming_distance,
    "jaccard_similarity": jaccard_similarity,
    "cosine_similarity_words": cosine_similarity_words,
    "longest_common_substring": longest_common_substring,
    "longest_common_subsequence": longest_common_subsequence,
    "damerau_levenshtein": damerau_levenshtein,
    "jaro_similarity": jaro_similarity,
    "jaro_winkler": jaro_winkler,
    "sorensen_dice": sorensen_dice,
    "normalized_levenshtein": normalized_levenshtein,
    "is_anagram": is_anagram,
    "is_palindrome": is_palindrome,
    "common_prefix": common_prefix,
}

STRING_METRICS_NL_PATTERNS = [
    (r'(?:hamming)\s+distance\s+(?:between|of)\s+["\']?(\w+)["\']?\s+(?:and)\s+["\']?(\w+)["\']?', 'hamming_distance("{0}", "{1}")'),
    (r'(?:jaccard)\s+similarity\s+(?:between|of)', None),
    (r'(?:jaro|jaro.winkler)\s+(?:similarity|distance)\s+(?:between|of)\s+["\']?(\w+)["\']?\s+(?:and)\s+["\']?(\w+)["\']?', 'jaro_winkler("{0}", "{1}")'),
    (r'(?:is|are)\s+["\']?(\w+)["\']?\s+(?:and)\s+["\']?(\w+)["\']?\s+anagrams?', 'is_anagram("{0}", "{1}")'),
    (r'(?:is)\s+["\']?(.+?)["\']?\s+(?:a\s+)?palindrome', 'is_palindrome("{0}")'),
    (r'(?:longest\s+)?common\s+(?:substring|subsequence)\s+(?:of|between)', None),
]
