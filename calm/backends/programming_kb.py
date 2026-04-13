"""
CALM Programming concepts knowledge backend — paradigms, principles, anti-patterns.

Models misstate SOLID, confuse patterns, give wrong definitions.
"""

from __future__ import annotations

_DATA_VERSION = "2025-01"

_SOLID = {
    "S": {
        "name": "Single Responsibility Principle",
        "definition": "A class should have only one reason to change",
        "violation": "God class that handles UI, business logic, and database",
        "fix": "Split into focused classes, each owning one concern",
    },
    "O": {
        "name": "Open/Closed Principle",
        "definition": "Open for extension, closed for modification",
        "violation": "Adding new behavior requires modifying existing code",
        "fix": "Use interfaces/abstract classes, strategy pattern",
    },
    "L": {
        "name": "Liskov Substitution Principle",
        "definition": "Subtypes must be substitutable for their base types",
        "violation": "Square extends Rectangle but breaks setWidth/setHeight",
        "fix": "Ensure subclass honors all contracts of the parent",
    },
    "I": {
        "name": "Interface Segregation Principle",
        "definition": "Clients should not depend on interfaces they don't use",
        "violation": "One fat interface with 20 methods, most unused by each client",
        "fix": "Split into small, focused interfaces",
    },
    "D": {
        "name": "Dependency Inversion Principle",
        "definition": "Depend on abstractions, not concretions",
        "violation": "High-level module directly instantiates low-level module",
        "fix": "Inject dependencies via constructor/interface",
    },
}

_PARADIGMS = {
    "imperative": {"description": "Step-by-step instructions (how to do it)", "languages": ["C", "Go", "Assembly"], "key_concept": "statements, control flow"},
    "declarative": {"description": "Describe what you want (not how)", "languages": ["SQL", "HTML", "CSS", "Prolog"], "key_concept": "expressions, constraints"},
    "object-oriented": {"description": "Organize code around objects with state and behavior", "languages": ["Java", "C#", "Python", "Ruby"], "key_concept": "encapsulation, inheritance, polymorphism"},
    "functional": {"description": "Compose pure functions, avoid shared state", "languages": ["Haskell", "Erlang", "Clojure", "F#"], "key_concept": "immutability, higher-order functions, no side effects"},
    "reactive": {"description": "Asynchronous data streams and propagation of change", "languages": ["RxJS", "Reactor", "Akka Streams"], "key_concept": "observables, backpressure"},
    "concurrent": {"description": "Multiple computations executing simultaneously", "models": ["threads (shared memory)", "actors (message passing)", "CSP (channels)", "async/await (coroutines)"]},
    "logic": {"description": "Express facts and rules, engine derives conclusions", "languages": ["Prolog", "Datalog"], "key_concept": "unification, backtracking"},
}

_ANTI_PATTERNS = {
    "god object": {"description": "One class that knows/does everything", "fix": "Split by responsibility"},
    "spaghetti code": {"description": "Tangled control flow, impossible to follow", "fix": "Extract functions, reduce nesting"},
    "golden hammer": {"description": "Using one familiar tool for everything", "fix": "Choose the right tool for each problem"},
    "premature optimization": {"description": "Optimizing before measuring", "fix": "Profile first, optimize bottlenecks"},
    "cargo cult": {"description": "Copying patterns without understanding why", "fix": "Understand the problem each pattern solves"},
    "magic numbers": {"description": "Unexplained numeric literals in code", "fix": "Named constants"},
    "shotgun surgery": {"description": "One change requires editing many files", "fix": "Move related logic together"},
    "feature envy": {"description": "Method uses more of another class's data than its own", "fix": "Move method to the class it envies"},
    "primitive obsession": {"description": "Using primitives instead of small objects", "fix": "Value objects (Money, Email, etc.)"},
    "callback hell": {"description": "Deeply nested callbacks", "fix": "Promises, async/await, reactive streams"},
    "n+1 query": {"description": "One query per item instead of batch", "fix": "Eager loading, JOIN, batch queries"},
    "stringly typed": {"description": "Using strings where types/enums should be used", "fix": "Enums, branded types, ADTs"},
}

_PRINCIPLES = {
    "DRY": {"full": "Don't Repeat Yourself", "description": "Every piece of knowledge should have a single source", "caveat": "Don't DRY too early — 3 is the threshold for extraction"},
    "KISS": {"full": "Keep It Simple, Stupid", "description": "Simpler solutions are easier to understand, maintain, debug"},
    "YAGNI": {"full": "You Ain't Gonna Need It", "description": "Don't build features until they're actually needed"},
    "separation of concerns": {"description": "Divide program into distinct sections, each addressing one concern"},
    "composition over inheritance": {"description": "Favor object composition over class inheritance for code reuse"},
    "law of demeter": {"description": "Don't talk to strangers — only call methods on immediate friends", "alias": "principle of least knowledge"},
    "tell don't ask": {"description": "Tell objects to do things instead of querying their state and deciding"},
    "fail fast": {"description": "Detect and report errors at the earliest opportunity"},
    "convention over configuration": {"description": "Sensible defaults reduce boilerplate (Rails, Spring Boot)"},
    "inversion of control": {"description": "Framework calls your code, not vice versa", "alias": "Hollywood principle"},
}


def solid_principle(letter: str) -> dict:
    """Get SOLID principle by letter (S, O, L, I, D)."""
    key = str(letter).upper().strip()[0]
    entry = _SOLID.get(key)
    if not entry:
        return {"error": f"Unknown: {letter}", "valid": list(_SOLID.keys())}
    return {"letter": key, **entry}


def all_solid() -> dict:
    """Get all 5 SOLID principles."""
    return {k: v["name"] for k, v in _SOLID.items()}


def paradigm_info(name: str) -> dict:
    """Get details about a programming paradigm."""
    key = str(name).lower().strip().replace("-", " ")
    entry = _PARADIGMS.get(key)
    if not entry:
        for k, v in _PARADIGMS.items():
            if key in k:
                return {"paradigm": k, **v}
        return {"error": f"Unknown: {name}", "valid": list(_PARADIGMS.keys())}
    return {"paradigm": key, **entry}


def anti_pattern(name: str) -> dict:
    """Get details about a software anti-pattern."""
    key = str(name).lower().strip()
    entry = _ANTI_PATTERNS.get(key)
    if not entry:
        for k, v in _ANTI_PATTERNS.items():
            if key in k or k in key:
                return {"anti_pattern": k, **v}
        return {"error": f"Unknown: {name}", "valid": list(_ANTI_PATTERNS.keys())}
    return {"anti_pattern": key, **entry}


def principle_info(name: str) -> dict:
    """Get details about a software principle."""
    key = str(name).lower().strip()
    entry = _PRINCIPLES.get(key)
    if not entry:
        for k, v in _PRINCIPLES.items():
            if key in k or k in key or key == v.get("alias", ""):
                return {"principle": k, **v}
        return {"error": f"Unknown: {name}", "valid": list(_PRINCIPLES.keys())}
    return {"principle": key, **entry}


def list_anti_patterns() -> list[str]:
    """List all known anti-patterns."""
    return sorted(_ANTI_PATTERNS.keys())


PROGRAMMING_FUNCTIONS = {
    "solid_principle": solid_principle,
    "all_solid": all_solid,
    "paradigm_info": paradigm_info,
    "anti_pattern": anti_pattern,
    "principle_info": principle_info,
    "list_anti_patterns": list_anti_patterns,
}

PROGRAMMING_NL_PATTERNS = [
    (r'(?:what is|explain)\s+(?:the\s+)?([SOLID])\s+(?:in|of|from)\s+SOLID', 'solid_principle("{0}")'),
    (r'(?:what is|explain)\s+SOLID', 'all_solid()'),
    (r'(?:what is|explain)\s+(?:the\s+)?(imperative|declarative|object.oriented|functional|reactive|concurrent|logic)\s+(?:paradigm|programming)', 'paradigm_info("{0}")'),
    (r'(?:what is|explain)\s+(?:the\s+)?(god object|spaghetti|golden hammer|cargo cult|n\+1|callback hell|shotgun surgery)', 'anti_pattern("{0}")'),
    (r'(?:what is|explain)\s+(?:the\s+)?(DRY|KISS|YAGNI|tell don.t ask|fail fast|law of demeter)', 'principle_info("{0}")'),
]
