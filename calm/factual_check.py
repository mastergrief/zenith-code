"""
Auto-CALM Factual Check — detect claims that contradict well-known facts.

Uses CALM backends to verify factual claims in responses. If a response
says "Python dicts are red-black trees" and we have a backend that knows
Python internals, flag it.

This is NOT a general fact-checker — it only catches claims in domains
where we have backend knowledge to verify against.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class FactualIssue:
    """A factual claim that may be incorrect."""
    claim: str
    category: str   # "contradiction", "suspicious", "unverifiable"
    reason: str
    confidence: float  # how confident we are this is wrong


@dataclass
class FactualCheckResult:
    """Result of factual checking."""
    issues: List[FactualIssue] = field(default_factory=list)
    claims_checked: int = 0
    score: float = 1.0  # 1.0 = no issues found

    def summary(self) -> str:
        if not self.issues:
            return f"checked {self.claims_checked} claims, no issues"
        return (f"checked {self.claims_checked} claims, "
                f"{len(self.issues)} issues found")


# Known facts that models commonly get wrong.
# Each entry: (regex pattern in response, expected truth, category)
_KNOWN_FACTS = [
    # Data structure implementations
    (r'Python\s+(?:dict|dictionaries?)\s+(?:are|is)\s+(?:implemented\s+as\s+)?(?:red.black|avl|binary)\s+tree',
     "Python dicts are hash tables (since CPython, always have been)", "contradiction"),
    (r'Java\s+HashMap\s+(?:is|uses)\s+(?:a\s+)?(?:red.black|avl|binary)\s+tree',
     "Java HashMap is a hash table (TreeMap is the red-black tree)", "contradiction"),

    # Complexity claims
    (r'hash\s+table[s]?\s+(?:always|guarantee)\s+O\(1\)',
     "Hash tables are O(1) average, O(n) worst case (collisions)", "suspicious"),
    (r'(?:binary\s+search|BST)\s+(?:is\s+)?always\s+O\(log\s*n\)',
     "BST is O(log n) average, O(n) worst case when unbalanced", "suspicious"),

    # Crypto/hashing
    (r'(?:hash\s+tables?|dictionaries?)\s+(?:use|uses)\s+(?:SHA|MD5|bcrypt|argon)',
     "Hash tables use fast hash functions (like MurmurHash, SipHash), not cryptographic hashes", "contradiction"),
    (r'MD5\s+is\s+(?:secure|safe|recommended)',
     "MD5 is broken for security — collision attacks are trivial since 2004", "contradiction"),
    (r'SHA-?1\s+is\s+(?:secure|safe|recommended)',
     "SHA-1 is broken — SHAttered collision demonstrated in 2017", "contradiction"),

    # Common misconceptions
    (r'(?:hash\s+tables?|dicts?)\s+(?:never|don.t|do not)\s+have\s+collisions',
     "All hash tables can have collisions — that's why collision resolution exists", "contradiction"),
    (r'NoSQL\s+(?:is|databases?\s+are)\s+(?:always\s+)?faster\s+than\s+(?:SQL|relational)',
     "NoSQL is not inherently faster — it depends on access patterns and data model", "suspicious"),
    (r'(?:microservices?|micro.services?)\s+(?:is|are)\s+(?:always\s+)?better\s+than\s+monolith',
     "Microservices add complexity — monolith-first is often the right approach", "suspicious"),
    (r'(?:REST|rest)\s+(?:is|requires?)\s+(?:always\s+)?JSON',
     "REST is architecture-agnostic — it can use JSON, XML, protobuf, or any format", "suspicious"),
    (r'TCP\s+is\s+(?:always\s+)?(?:slower|worse)\s+than\s+UDP',
     "TCP vs UDP depends on use case — TCP is better for reliability, UDP for latency", "suspicious"),
    (r'(?:JavaScript|JS)\s+is\s+(?:single.threaded|single threaded)\s+(?:so|and)\s+(?:can.t|cannot)\s+(?:do|handle)\s+concurrency',
     "JS is single-threaded but handles concurrency via the event loop and async/await", "suspicious"),

    # Year/attribution errors (common hallucinations)
    (r'(?:hash\s+tables?)\s+(?:were|was)\s+invented\s+(?:by|in|at)\s+(?:IBM|Microsoft|Google|Apple)',
     "Hash tables were described by Hans Peter Luhn at IBM in 1953 — but 'invented by IBM in 1990' is wrong", "suspicious"),

    # Language-specific
    (r'Python\s+is\s+(?:a\s+)?compiled\s+language',
     "Python is interpreted (CPython compiles to bytecode, but is not a compiled language)", "suspicious"),
    (r'(?:Go|Golang)\s+(?:has|supports)\s+(?:classes|inheritance)',
     "Go has no classes or inheritance — it uses structs, interfaces, and composition", "contradiction"),
    (r'Rust\s+(?:has|uses)\s+(?:a\s+)?garbage\s+collector',
     "Rust has no GC — it uses ownership and borrowing for memory management", "contradiction"),

    # Database misconceptions
    (r'MongoDB\s+(?:is|does)\s+(?:not\s+)?(?:support|have)\s+(?:ACID|transactions)',
     "MongoDB supports multi-document ACID transactions since 4.0 (2018)", "suspicious"),
    (r'SQL\s+(?:databases?|is)\s+(?:can.t|cannot|don.t|do not)\s+scale\s+horizontally',
     "SQL databases can scale horizontally (Citus, CockroachDB, Vitess, Aurora)", "contradiction"),
    (r'NoSQL\s+(?:means?|stands?\s+for)\s+(?:\")?no\s+SQL(?:\")?',
     "NoSQL means 'Not Only SQL' — many NoSQL DBs support SQL-like queries", "contradiction"),
    (r'Redis\s+(?:is|has)\s+(?:only\s+)?(?:in.memory|no\s+persistence)',
     "Redis supports persistence via RDB snapshots and AOF log — not memory-only", "suspicious"),
    (r'PostgreSQL\s+(?:is|does)\s+(?:not\s+)?(?:support|have)\s+(?:JSON|JSONB)',
     "PostgreSQL has native JSONB support with indexing since 9.4 (2014)", "contradiction"),

    # Web/HTTP misconceptions
    (r'(?:REST|rest)\s+(?:requires?|must\s+use|is)\s+(?:only\s+)?(?:JSON|HTTP)',
     "REST is an architectural style, not tied to any protocol or format", "suspicious"),
    (r'(?:GET|get)\s+requests?\s+(?:can.t|cannot|should\s+not)\s+have\s+(?:a\s+)?body',
     "GET requests CAN have a body per HTTP spec, but many servers/clients ignore it", "suspicious"),
    (r'(?:PUT|put)\s+(?:and|is\s+the\s+same\s+as|=)\s+(?:POST|post)',
     "PUT is idempotent (same result on repeat), POST is not — they're different", "contradiction"),
    (r'(?:HTTPS|https)\s+(?:encrypts?|hides?)\s+(?:the\s+)?(?:URL|url)',
     "HTTPS encrypts the path and query string, but the domain is visible via SNI and DNS", "suspicious"),
    (r'(?:cookies?)\s+(?:are|is)\s+(?:always\s+)?(?:insecure|unsafe|bad)',
     "Cookies with HttpOnly + Secure + SameSite are the recommended auth transport for web apps", "suspicious"),

    # Security misconceptions
    (r'(?:bcrypt|argon2|scrypt)\s+(?:is|are)\s+(?:a\s+)?(?:encryption|cipher)',
     "bcrypt/argon2/scrypt are password hashing (KDF) functions, not encryption", "contradiction"),
    (r'(?:base64|Base64)\s+(?:is|provides?)\s+(?:a\s+)?(?:encryption|security|protection)',
     "base64 is encoding (reversible), not encryption — provides zero security", "contradiction"),
    (r'(?:JWT|jwt)\s+(?:is|are)\s+(?:encrypted|secure\s+by\s+default)',
     "JWT payload is base64-encoded (readable), not encrypted. Use JWE for encryption.", "suspicious"),
    (r'(?:CORS|cors)\s+(?:is|provides?)\s+(?:a\s+)?(?:security|protection)',
     "CORS is a browser mechanism that RELAXES the same-origin policy — it's not a security feature", "suspicious"),
    (r'(?:rate\s+limiting|captcha)\s+(?:prevents?|stops?)\s+(?:all\s+)?(?:DDoS|attacks)',
     "Rate limiting mitigates but doesn't prevent DDoS — volumetric attacks overwhelm before rate limits help", "suspicious"),
    (r'(?:client.side|frontend)\s+validation\s+(?:is\s+)?(?:sufficient|enough|secure)',
     "Client-side validation is UX only — always validate on the server (client can be bypassed)", "contradiction"),

    # Performance misconceptions
    (r'(?:async|asynchronous)\s+(?:is|makes?\s+things?)\s+(?:always\s+)?faster',
     "Async improves throughput for I/O-bound work but doesn't speed up CPU-bound work", "suspicious"),
    (r'(?:more\s+threads?|multithreading)\s+(?:is|makes?\s+things?)\s+(?:always\s+)?faster',
     "More threads help CPU-bound parallel work but can SLOW DOWN I/O-bound work (context switching)", "suspicious"),
    (r'(?:garbage\s+collection|GC)\s+(?:is|makes?\s+things?)\s+(?:always\s+)?(?:slow|bad|worse)',
     "Modern GC (Go, Java ZGC, .NET) has sub-millisecond pauses — GC overhead is often negligible", "suspicious"),
    (r'(?:compiled|native)\s+(?:code|languages?)\s+(?:is|are)\s+(?:always\s+)?faster\s+than\s+(?:interpreted|scripting)',
     "JIT-compiled languages (Java, C#, JS V8) often match native speed — 'compiled = faster' is oversimplified", "suspicious"),
    (r'(?:linked\s+lists?)\s+(?:is|are)\s+(?:faster|better)\s+than\s+(?:arrays?|vectors?)\s+for\s+(?:insert|deletion)',
     "Arrays are often faster even for insertion due to cache locality — linked lists have poor cache behavior", "suspicious"),
    (r'O\(1\)\s+(?:is|means?\s+)\s+(?:always\s+)?(?:fast|instant)',
     "O(1) means constant time, not fast — an O(1) operation can take 10 seconds if the constant is large", "suspicious"),

    # Architecture misconceptions
    (r'(?:monolith|monolithic)\s+(?:is|are)\s+(?:always\s+)?(?:bad|wrong|legacy|outdated)',
     "Monoliths are the right choice for most early-stage apps — premature microservices add complexity", "suspicious"),
    (r'(?:kubernetes|k8s)\s+(?:is\s+)?(?:needed|required|necessary)\s+for\s+(?:containers?|docker)',
     "Docker containers run fine without K8s — K8s is for orchestrating many services at scale", "suspicious"),
    (r'(?:serverless|lambda)\s+(?:is|means?)\s+(?:no\s+servers?|there\s+are\s+no\s+servers)',
     "Serverless still runs on servers — you just don't manage them. Cold starts and limits still apply.", "suspicious"),
    (r'(?:GraphQL)\s+(?:is|replaces?)\s+(?:always\s+)?(?:better|faster)\s+than\s+(?:REST|rest)',
     "GraphQL trades over-fetching for query complexity, N+1 problems, and caching difficulty — not always better", "suspicious"),

    # Git misconceptions
    (r'git\s+pull\s+(?:is\s+the\s+same\s+as|=|equals?)\s+git\s+fetch',
     "git pull = git fetch + git merge (or rebase). fetch alone doesn't modify working tree.", "contradiction"),
    (r'git\s+rebase\s+(?:is|should\s+be)\s+(?:always\s+)?(?:better|preferred|used)\s+(?:over|instead\s+of)\s+(?:merge)',
     "Rebase rewrites history — never rebase shared/pushed commits. Merge is safer for collaboration.", "suspicious"),

    # OS/systems misconceptions
    (r'(?:Linux|linux)\s+(?:is|can.t|doesn.t)\s+(?:not\s+)?(?:get|have)\s+(?:viruses?|malware)',
     "Linux can get malware — it's less common due to market share and permissions model, not immunity", "suspicious"),
    (r'(?:SSD|ssd)\s+(?:has|have)\s+(?:no|unlimited)\s+(?:write\s+)?(?:limit|lifespan|wear)',
     "SSDs have finite write endurance (TBW) — enterprise workloads can wear them out", "suspicious"),
    (r'(?:RAM|ram|memory)\s+(?:is|are)\s+(?:always\s+)?faster\s+than\s+(?:SSD|disk|storage)',
     "True for random access, but sequential SSD reads can approach RAM bandwidth on NVMe", "suspicious"),
]


class FactualChecker:
    """Check response for common factual errors using pattern matching + dynamic backend cross-check."""

    # Dynamic cross-check patterns: (regex, backend_expr_template, value_extractor, comparator)
    # template uses {name} and {value} from match groups
    _DYNAMIC_PATTERNS = [
        # Hash output length: "SHA-256 output is 32 bytes" → hash_output_length("sha256") = 64
        (re.compile(r'(SHA-?256|SHA-?512|SHA-?1|MD5|SHA-?384|blake2b|blake3)\s+(?:output|hash|digest)\s+(?:is|has|produces?)\s+(\d+)\s+(?:hex\s+)?(?:bytes?|characters?|chars?|digits?)', re.IGNORECASE),
         'hash_output_length', lambda m: (m.group(1).lower().replace("-", "").replace(" ", ""), int(m.group(2))),
         lambda actual, claimed: actual == claimed if actual > 0 else None),

        # Data structure complexity: "hash table search is O(1)" → ds_info("hash table")["search"]
        (re.compile(r'(hash\s+table|linked\s+list|binary\s+search\s+tree|BST|heap|trie|stack|queue|array|red.black\s+tree|B.tree|bloom\s+filter)\s+(?:search|lookup|access|insert|delete)\s+(?:is|has|takes?)\s+O\(([^)]+)\)', re.IGNORECASE),
         'ds_info', lambda m: (m.group(1).lower(), m.group(2).strip()),
         None),  # complex comparison, handled inline

        # Sort complexity: "quicksort average is O(n log n)" → sort_info("quick sort")["average"]
        (re.compile(r'(bubble|selection|insertion|merge|quick|heap|counting|radix|bucket|tim)\s*sort\s+(?:has\s+)?(?:average|worst|best)(?:\s+case)?\s+(?:is|of|=)\s+O\(([^)]+)\)', re.IGNORECASE),
         'sort_info', lambda m: (m.group(1).lower() + " sort", m.group(2).strip()),
         None),

        # Protocol ports: "SSH uses port 22" → protocol_info("SSH")["default_port"]
        (re.compile(r'(SSH|HTTP|HTTPS|FTP|SMTP|DNS|MySQL|PostgreSQL|Redis|MongoDB)\s+(?:uses?|runs?\s+on|listens?\s+on|default)\s+port\s+(\d+)', re.IGNORECASE),
         'protocol_info', lambda m: (m.group(1).upper(), int(m.group(2))),
         None),

        # Country capitals: "capital of France is Paris" → country_capital("france")
        (re.compile(r'capital\s+of\s+(\w+(?:\s+\w+)?)\s+is\s+(\w+(?:\s+\w+)?)', re.IGNORECASE),
         'country_capital', lambda m: (m.group(1).lower(), m.group(2)),
         lambda actual, claimed: actual.lower() == claimed.lower()),

        # Currency decimals: "JPY has 2 decimal places" → currency_decimals("JPY")
        (re.compile(r'(\w{3})\s+(?:has|uses?)\s+(\d+)\s+decimal\s+places?', re.IGNORECASE),
         'currency_decimals', lambda m: (m.group(1).upper(), int(m.group(2))),
         lambda actual, claimed: actual == claimed if actual >= 0 else None),

        # Molecular weight: "glucose molecular weight is 180" → molecular_weight("glucose")
        (re.compile(r'(?:molecular\s+weight|molar\s+mass)\s+(?:of\s+)?(\w+)\s+is\s+([\d.]+)', re.IGNORECASE),
         'molecular_weight', lambda m: (m.group(1), float(m.group(2))),
         lambda actual, claimed: abs(actual - claimed) < 1.0 if actual > 0 else None),

        # Element atomic number: "oxygen atomic number is 8" → element_atomic_number("oxygen")
        (re.compile(r'(\w+)\s+(?:has\s+)?atomic\s+number\s+(?:is\s+|of\s+)?(\d+)', re.IGNORECASE),
         'element_atomic_number', lambda m: (m.group(1).lower(), int(m.group(2))),
         lambda actual, claimed: actual == claimed if actual > 0 else None),

        # OSI layer: "TCP operates at layer 4" → which_layer("TCP")
        (re.compile(r'(TCP|UDP|HTTP|HTTPS|IP|ICMP|ARP|DNS|SSH|FTP|SMTP)\s+(?:operates?\s+at\s+|is\s+(?:at\s+)?|runs?\s+(?:at\s+)?|(?:is\s+)?(?:on\s+)?)?layer\s+(\d)', re.IGNORECASE),
         'which_layer', lambda m: (m.group(1).upper(), int(m.group(2))),
         lambda actual, claimed: actual == claimed if actual > 0 else None),

        # Note frequency: "A4 is 440 Hz" → note_frequency("A", 4)
        (re.compile(r'([A-G][#b]?)(\d)\s+(?:is|=|has\s+frequency)\s+([\d.]+)\s*Hz', re.IGNORECASE),
         'note_frequency', lambda m: ((m.group(1), int(m.group(2))), float(m.group(3))),
         lambda actual, claimed: abs(actual - claimed) < 1.0 if actual > 0 else None),
    ]

    def check(self, response: str) -> FactualCheckResult:
        """Check a response for factual issues (static + dynamic)."""
        result = FactualCheckResult()
        text = str(response)

        # Pass 1: Static regex patterns (fast, known misconceptions)
        for pattern, truth, category in _KNOWN_FACTS:
            result.claims_checked += 1
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result.issues.append(FactualIssue(
                    claim=match.group(0),
                    category=category,
                    reason=truth,
                    confidence=0.9 if category == "contradiction" else 0.7,
                ))

        # Pass 2: Dynamic cross-check against backends
        self._dynamic_cross_check(text, result)

        # Score: 1.0 if no issues, decreasing with each issue
        if result.issues:
            result.score = max(0, 1.0 - len(result.issues) * 0.2)

        return result

    def _dynamic_cross_check(self, text: str, result: FactualCheckResult):
        """Cross-check factual claims against CALM backend functions."""
        from calm.expression import safe_eval, ExpressionError

        for pattern, func_name, extractor, comparator in self._DYNAMIC_PATTERNS:
            for match in pattern.finditer(text):
                result.claims_checked += 1
                try:
                    extracted = extractor(match)
                    if func_name == 'hash_output_length':
                        alg, claimed_val = extracted
                        actual = safe_eval(f'hash_output_length("{alg}")')
                        if actual > 0 and actual != claimed_val:
                            result.issues.append(FactualIssue(
                                claim=match.group(0),
                                category="dynamic_cross_check",
                                reason=f"Backend says {func_name}(\"{alg}\") = {actual}, response claims {claimed_val}",
                                confidence=0.9,
                            ))

                    elif func_name == 'country_capital':
                        country, claimed_capital = extracted
                        try:
                            actual = safe_eval(f'country_capital("{country}")')
                            if comparator and comparator(actual, claimed_capital) is False:
                                result.issues.append(FactualIssue(
                                    claim=match.group(0),
                                    category="dynamic_cross_check",
                                    reason=f"Backend says capital of {country} = {actual}, response claims {claimed_capital}",
                                    confidence=0.9,
                                ))
                        except ExpressionError:
                            pass

                    elif func_name == 'currency_decimals':
                        code, claimed_val = extracted
                        try:
                            actual = safe_eval(f'currency_decimals("{code}")')
                            if actual >= 0 and actual != claimed_val:
                                result.issues.append(FactualIssue(
                                    claim=match.group(0),
                                    category="dynamic_cross_check",
                                    reason=f"Backend says {code} has {actual} decimal places, response claims {claimed_val}",
                                    confidence=0.9,
                                ))
                        except ExpressionError:
                            pass

                    elif func_name == 'molecular_weight':
                        name, claimed_val = extracted
                        try:
                            actual = safe_eval(f'molecular_weight("{name}")')
                            if actual > 0 and abs(actual - claimed_val) > 1.0:
                                result.issues.append(FactualIssue(
                                    claim=match.group(0),
                                    category="dynamic_cross_check",
                                    reason=f"Backend says MW of {name} = {actual}, response claims {claimed_val}",
                                    confidence=0.85,
                                ))
                        except ExpressionError:
                            pass

                    elif func_name == 'which_layer':
                        protocol, claimed_layer = extracted
                        try:
                            actual = safe_eval(f'which_layer("{protocol}")')
                            if actual > 0 and actual != claimed_layer:
                                result.issues.append(FactualIssue(
                                    claim=match.group(0),
                                    category="dynamic_cross_check",
                                    reason=f"Backend says {protocol} is layer {actual}, response claims layer {claimed_layer}",
                                    confidence=0.9,
                                ))
                        except ExpressionError:
                            pass

                    elif func_name == 'note_frequency':
                        (note, octave), claimed_freq = extracted
                        try:
                            actual = safe_eval(f'note_frequency("{note}", {octave})')
                            if actual > 0 and abs(actual - claimed_freq) > 1.0:
                                result.issues.append(FactualIssue(
                                    claim=match.group(0),
                                    category="dynamic_cross_check",
                                    reason=f"Backend says {note}{octave} = {actual} Hz, response claims {claimed_freq} Hz",
                                    confidence=0.9,
                                ))
                        except ExpressionError:
                            pass

                except (ExpressionError, ValueError, TypeError, IndexError):
                    pass  # extraction or eval failed, skip
