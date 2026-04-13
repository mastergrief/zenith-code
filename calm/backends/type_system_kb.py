"""
CALM Type system knowledge backend — type theory, generics, variance, common types.

Models confuse covariance/contravariance, mix up nominal vs structural typing.
"""

from __future__ import annotations

_DATA_VERSION = "2025-01"

_TYPE_SYSTEMS = {
    "static": {"description": "Types checked at compile time", "languages": ["Java", "C", "Rust", "TypeScript", "Go", "Haskell"], "pros": ["catch errors early", "IDE support", "documentation"], "cons": ["more verbose", "slower iteration"]},
    "dynamic": {"description": "Types checked at runtime", "languages": ["Python", "JavaScript", "Ruby", "PHP", "Lua"], "pros": ["less boilerplate", "faster prototyping", "flexibility"], "cons": ["runtime errors", "harder refactoring"]},
    "gradual": {"description": "Mix of static and dynamic typing", "languages": ["TypeScript", "Python (with type hints)", "Dart", "C#"], "description_2": "Add types incrementally where they help"},
    "strong": {"description": "No implicit type coercion (or very limited)", "languages": ["Python", "Rust", "Haskell", "Java"], "example": "'1' + 2 → TypeError in Python"},
    "weak": {"description": "Implicit type coercion allowed", "languages": ["JavaScript", "PHP", "C"], "example": "'1' + 2 → '12' in JavaScript, 3 in PHP"},
    "nominal": {"description": "Types compatible only if same name/declaration", "languages": ["Java", "C#", "Rust"], "example": "class Foo != class Bar even if identical structure"},
    "structural": {"description": "Types compatible if same structure (duck typing)", "languages": ["TypeScript", "Go (interfaces)", "OCaml"], "example": "any {x: number} is compatible with Point if Point has x: number"},
    "dependent": {"description": "Types can depend on values", "languages": ["Idris", "Agda", "Coq"], "example": "Vector(n) where n is the length — type encodes the size"},
    "linear": {"description": "Values must be used exactly once", "languages": ["Rust (ownership)", "Linear Haskell"], "benefit": "compile-time memory safety without GC"},
    "algebraic": {"description": "Types built from sum (|) and product (&) types", "languages": ["Haskell", "Rust", "OCaml", "TypeScript"], "sum": "enum/tagged union: A | B", "product": "struct/record: A & B"},
}

_VARIANCE = {
    "covariant": {"symbol": "out/+", "rule": "subtype relationship preserved: if Dog <: Animal, then List<Dog> <: List<Animal>", "use": "read-only (producer)", "example": "Kotlin: List<out T>", "safe_for": "returning values"},
    "contravariant": {"symbol": "in/-", "rule": "subtype relationship reversed: if Dog <: Animal, then Consumer<Animal> <: Consumer<Dog>", "use": "write-only (consumer)", "example": "Kotlin: Comparable<in T>", "safe_for": "accepting values"},
    "invariant": {"symbol": "none", "rule": "no subtype relationship: List<Dog> is NOT related to List<Animal>", "use": "read+write", "example": "Java: List<T> (mutable), Rust: Vec<T>", "reason": "both covariant and contravariant would be unsound for mutable containers"},
    "bivariant": {"symbol": "+/-", "rule": "both co- and contra-variant (unsound, but TypeScript uses for function params for pragmatism)", "note": "generally a type system bug/pragmatic compromise"},
}

_COMMON_TYPES = {
    "Option/Maybe": {"description": "Value that might be absent. Prevents null pointer errors.", "languages": {"Rust": "Option<T>", "Haskell": "Maybe a", "Java": "Optional<T>", "Kotlin": "T?", "Swift": "T?", "TypeScript": "T | undefined"}},
    "Result/Either": {"description": "Success value OR error. Replaces exceptions for expected failures.", "languages": {"Rust": "Result<T, E>", "Haskell": "Either a b", "Go": "(T, error)", "TypeScript": "T | Error"}},
    "Tuple": {"description": "Fixed-size, heterogeneous collection.", "languages": {"Python": "tuple[int, str]", "Rust": "(i32, String)", "TypeScript": "[number, string]", "Haskell": "(Int, String)"}},
    "Union/Sum": {"description": "Value is one of several types.", "languages": {"TypeScript": "A | B", "Python": "Union[A, B] or A | B", "Rust": "enum { A(...), B(...) }", "Haskell": "data Foo = A Int | B String"}},
    "Intersection": {"description": "Value satisfies multiple types simultaneously.", "languages": {"TypeScript": "A & B", "Scala": "A with B"}},
    "Generic": {"description": "Parameterized type. Same logic for different types.", "languages": {"Java": "List<T>", "Rust": "Vec<T>", "TypeScript": "Array<T>", "Go": "[]T (since 1.18)", "Python": "list[T]"}},
    "Phantom": {"description": "Type parameter used only for compile-time checking, not at runtime.", "languages": {"Rust": "PhantomData<T>", "Haskell": "Proxy a"}, "use": "unit safety (Meters vs Feet), state machines"},
    "Newtype": {"description": "Wrapper type with zero runtime cost. Distinct type from the wrapped value.", "languages": {"Haskell": "newtype Foo = Foo Int", "Rust": "struct Foo(i32);", "TypeScript": "branded types"}},
}


def type_system(name: str) -> dict:
    """Get details about a type system characteristic."""
    key = str(name).lower().strip()
    entry = _TYPE_SYSTEMS.get(key)
    if not entry:
        return {"error": f"Unknown: {name}", "valid": list(_TYPE_SYSTEMS.keys())}
    return {"system": key, **entry}


def variance_info(kind: str) -> dict:
    """Get details about type variance."""
    key = str(kind).lower().strip()
    entry = _VARIANCE.get(key)
    if not entry:
        return {"error": f"Unknown: {kind}", "valid": list(_VARIANCE.keys())}
    return {"variance": key, **entry}


def common_type(name: str) -> dict:
    """Get details about a common type pattern."""
    key = str(name).strip()
    for k, v in _COMMON_TYPES.items():
        if key.lower() in k.lower():
            return {"type": k, **v}
    return {"error": f"Unknown: {name}", "valid": list(_COMMON_TYPES.keys())}


def static_vs_dynamic() -> dict:
    """Compare static and dynamic typing."""
    return {"static": _TYPE_SYSTEMS["static"], "dynamic": _TYPE_SYSTEMS["dynamic"]}


def nominal_vs_structural() -> dict:
    """Compare nominal and structural typing."""
    return {"nominal": _TYPE_SYSTEMS["nominal"], "structural": _TYPE_SYSTEMS["structural"]}


def strong_vs_weak() -> dict:
    """Compare strong and weak typing."""
    return {"strong": _TYPE_SYSTEMS["strong"], "weak": _TYPE_SYSTEMS["weak"]}


TYPE_SYSTEM_FUNCTIONS = {
    "type_system": type_system,
    "variance_info": variance_info,
    "common_type": common_type,
    "static_vs_dynamic": static_vs_dynamic,
    "nominal_vs_structural": nominal_vs_structural,
    "strong_vs_weak": strong_vs_weak,
}

TYPE_SYSTEM_NL_PATTERNS = [
    (r'(?:what is|explain)\s+(static|dynamic|gradual|strong|weak|nominal|structural|dependent|linear|algebraic)\s+typing', 'type_system("{0}")'),
    (r'(?:what is|explain)\s+(covariant|contravariant|invariant|bivariant)', 'variance_info("{0}")'),
    (r'(?:what is|explain)\s+(Option|Maybe|Result|Either|Tuple|Union|Intersection|Generic|Phantom|Newtype)\s+(?:type|pattern)', 'common_type("{0}")'),
    (r'(?:compare|difference|vs)\s+static\s+(?:and|vs)\s+dynamic\s+typing', 'static_vs_dynamic()'),
    (r'(?:compare|difference|vs)\s+nominal\s+(?:and|vs)\s+structural\s+typing', 'nominal_vs_structural()'),
    (r'(?:compare|difference|vs)\s+strong\s+(?:and|vs)\s+weak\s+typing', 'strong_vs_weak()'),
]
