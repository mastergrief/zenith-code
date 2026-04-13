"""
CALM Compiler/language theory knowledge backend — compilation stages, grammars, type inference.

Models confuse lexing vs parsing, hallucinate grammar types, mix up compilation stages.
"""

from __future__ import annotations

_DATA_VERSION = "2025-01"

_STAGES = {
    "lexing": {"alias": "tokenization", "input": "source code (characters)", "output": "token stream", "does": "breaks source into tokens (keywords, identifiers, literals, operators)", "tools": ["flex", "re2c", "hand-written DFA"]},
    "parsing": {"input": "token stream", "output": "AST (Abstract Syntax Tree)", "does": "checks syntactic structure, builds tree", "types": ["recursive descent (LL)", "LR (bottom-up)", "PEG", "Earley"], "tools": ["bison", "ANTLR", "tree-sitter", "pest"]},
    "semantic analysis": {"input": "AST", "output": "annotated AST", "does": "type checking, name resolution, scope analysis", "catches": ["type errors", "undefined variables", "scope violations"]},
    "IR generation": {"input": "annotated AST", "output": "intermediate representation (IR)", "examples": ["LLVM IR", "JVM bytecode", "three-address code"], "purpose": "machine-independent optimization target"},
    "optimization": {"input": "IR", "output": "optimized IR", "types": ["constant folding", "dead code elimination", "loop unrolling", "inlining", "register allocation", "vectorization"], "levels": ["-O0 (none)", "-O1 (basic)", "-O2 (moderate)", "-O3 (aggressive)", "-Os (size)"]},
    "code generation": {"input": "optimized IR", "output": "machine code or bytecode", "does": "instruction selection, register allocation, scheduling"},
    "linking": {"input": "object files", "output": "executable", "types": ["static linking (at build time)", "dynamic linking (at runtime)"], "tools": ["ld", "lld", "mold"]},
}

_GRAMMAR_TYPES = {
    "regular": {"chomsky": "Type 3", "recognized_by": "finite automaton (DFA/NFA)", "expressed_by": "regular expressions", "examples": ["identifiers", "number literals", "keywords"], "limitation": "can't match nested parentheses"},
    "context-free": {"chomsky": "Type 2", "recognized_by": "pushdown automaton (PDA)", "expressed_by": "BNF/EBNF", "examples": ["most programming language syntax", "balanced parentheses", "arithmetic expressions"], "limitation": "can't express 'declare before use'"},
    "context-sensitive": {"chomsky": "Type 1", "recognized_by": "linear-bounded automaton", "examples": ["type checking constraints", "C's typedef parsing"]},
    "recursively enumerable": {"chomsky": "Type 0", "recognized_by": "Turing machine", "note": "too powerful for practical use"},
}

_PARSING_STRATEGIES = {
    "recursive descent": {"direction": "top-down (LL)", "implementation": "hand-written functions, one per grammar rule", "pros": ["simple", "good error messages", "easy to understand"], "cons": ["no left recursion", "can be inefficient without memoization"], "used_by": ["GCC", "Clang", "Go", "Rust"]},
    "LL(k)": {"direction": "top-down", "description": "Left-to-right, Leftmost derivation, k tokens lookahead", "k_typical": "1 (LL(1))", "tools": ["ANTLR (LL(*))", "hand-written"]},
    "LR(k)": {"direction": "bottom-up", "description": "Left-to-right, Rightmost derivation (reversed), k tokens lookahead", "variants": ["SLR", "LALR(1)", "CLR"], "tools": ["bison/yacc (LALR(1))"], "pros": ["handles larger class of grammars", "no left-recursion problem"]},
    "PEG": {"full": "Parsing Expression Grammar", "description": "Ordered choice (first match wins). No ambiguity by construction.", "tools": ["pest (Rust)", "PEG.js", "packrat parsers"], "pros": ["no ambiguity", "composable"], "cons": ["ordered choice can be surprising", "left recursion needs special handling"]},
    "Earley": {"direction": "general", "description": "Parse any context-free grammar in O(n³), unambiguous in O(n²)", "pros": ["handles all CFGs", "good for ambiguous grammars"], "cons": ["slower for unambiguous grammars"]},
    "Pratt": {"alias": "top-down operator precedence", "description": "Elegant way to parse expressions with precedence and associativity", "pros": ["simple", "efficient", "extensible"], "used_by": ["many expression parsers", "jq", "Lua"]},
}

_EXECUTION_MODELS = {
    "compiled": {"description": "Source → machine code ahead of time", "examples": ["C", "C++", "Rust", "Go"], "pros": ["fast execution", "standalone binary"], "cons": ["platform-specific", "slower iteration"]},
    "interpreted": {"description": "Source executed line by line at runtime", "examples": ["Python (CPython)", "Ruby", "PHP", "Bash"], "pros": ["fast iteration", "portable source"], "cons": ["slow execution"]},
    "bytecode + VM": {"description": "Source → bytecode, executed by virtual machine", "examples": ["Java (JVM)", "C# (CLR)", "Python (.pyc)", "Lua"], "pros": ["portability", "optimization opportunities (JIT)"], "cons": ["startup overhead"]},
    "JIT": {"full": "Just-In-Time compilation", "description": "Compile hot code paths at runtime", "examples": ["V8 (JavaScript)", "HotSpot (Java)", "PyPy (Python)", "LuaJIT"], "pros": ["near-native speed", "runtime optimization"], "cons": ["warmup time", "memory overhead"]},
    "transpiled": {"description": "Source language → different source language", "examples": ["TypeScript → JavaScript", "Kotlin → JavaScript", "Sass → CSS"], "pros": ["use modern features", "target existing runtimes"]},
}


def compilation_stage(name: str) -> dict:
    """Get details about a compilation stage."""
    key = str(name).lower().strip()
    for k, v in _STAGES.items():
        if key in k or k in key:
            return {"stage": k, **v}
    return {"error": f"Unknown: {name}", "valid": list(_STAGES.keys())}


def grammar_type(name: str) -> dict:
    """Get details about a grammar type in the Chomsky hierarchy."""
    key = str(name).lower().strip()
    for k, v in _GRAMMAR_TYPES.items():
        if key in k:
            return {"type": k, **v}
    return {"error": f"Unknown: {name}", "valid": list(_GRAMMAR_TYPES.keys())}


def parsing_strategy(name: str) -> dict:
    """Get details about a parsing strategy."""
    key = str(name).lower().strip()
    for k, v in _PARSING_STRATEGIES.items():
        if key in k.lower() or k.lower() in key:
            return {"strategy": k, **v}
    return {"error": f"Unknown: {name}", "valid": list(_PARSING_STRATEGIES.keys())}


def execution_model(name: str) -> dict:
    """Get details about a language execution model."""
    key = str(name).lower().strip()
    for k, v in _EXECUTION_MODELS.items():
        if key in k.lower() or k.lower() in key:
            return {"model": k, **v}
    return {"error": f"Unknown: {name}", "valid": list(_EXECUTION_MODELS.keys())}


def compiled_vs_interpreted() -> dict:
    """Compare compiled vs interpreted execution."""
    return {"compiled": _EXECUTION_MODELS["compiled"], "interpreted": _EXECUTION_MODELS["interpreted"]}


def list_optimization_types() -> list[str]:
    """List common compiler optimization types."""
    return _STAGES["optimization"]["types"]


def lexing_vs_parsing() -> dict:
    """Compare lexing and parsing stages."""
    return {"lexing": _STAGES["lexing"], "parsing": _STAGES["parsing"]}


COMPILER_FUNCTIONS = {
    "compilation_stage": compilation_stage,
    "grammar_type": grammar_type,
    "parsing_strategy": parsing_strategy,
    "execution_model": execution_model,
    "compiled_vs_interpreted": compiled_vs_interpreted,
    "list_optimization_types": list_optimization_types,
    "lexing_vs_parsing": lexing_vs_parsing,
}

COMPILER_NL_PATTERNS = [
    (r'(?:what is|explain)\s+(?:the\s+)?(lexing|parsing|semantic analysis|IR generation|optimization|code generation|linking)\s+stage', 'compilation_stage("{0}")'),
    (r'(?:what is|explain)\s+(?:a\s+)?(regular|context.free|context.sensitive)\s+grammar', 'grammar_type("{0}")'),
    (r'(?:what is|explain)\s+(recursive descent|LL|LR|PEG|Earley|Pratt)\s+(?:parsing|parser)', 'parsing_strategy("{0}")'),
    (r'(?:what is|explain)\s+(compiled|interpreted|JIT|bytecode|transpiled)', 'execution_model("{0}")'),
    (r'(?:compare|difference|vs)\s+compiled\s+(?:and|vs)\s+interpreted', 'compiled_vs_interpreted()'),
    (r'(?:compare|difference|vs)\s+lexing\s+(?:and|vs)\s+parsing', 'lexing_vs_parsing()'),
]
