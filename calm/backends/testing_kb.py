"""
CALM Testing knowledge backend — test types, strategies, patterns, coverage.

Models confuse unit vs integration, hallucinate testing terminology.
"""

from __future__ import annotations

_DATA_VERSION = "2025-01"

_TEST_TYPES = {
    "unit": {"scope": "single function/method", "speed": "milliseconds", "isolation": "full (mocks/stubs)", "who": "developer", "ratio": "~70% of tests", "frameworks": {"python": "pytest", "javascript": "Jest/Vitest", "java": "JUnit", "rust": "cargo test", "go": "testing"}},
    "integration": {"scope": "multiple components together", "speed": "seconds", "isolation": "partial (real DB, mock external)", "who": "developer", "ratio": "~20% of tests"},
    "e2e": {"scope": "full application flow", "speed": "minutes", "isolation": "none (real everything)", "who": "QA/developer", "ratio": "~10% of tests", "frameworks": {"web": "Playwright/Cypress", "mobile": "Appium", "api": "Postman/Newman"}},
    "smoke": {"scope": "critical paths only", "speed": "seconds-minutes", "purpose": "verify deployment didn't break basics", "when": "after every deploy"},
    "regression": {"scope": "previously-broken functionality", "purpose": "ensure fixed bugs stay fixed", "when": "every CI run"},
    "performance": {"scope": "response time, throughput, resource usage", "types": ["load test", "stress test", "spike test", "soak test"], "frameworks": ["k6", "JMeter", "Gatling", "Locust"]},
    "security": {"scope": "vulnerabilities", "types": ["SAST (static)", "DAST (dynamic)", "SCA (dependencies)", "penetration testing"], "frameworks": ["OWASP ZAP", "Burp Suite", "Snyk", "SonarQube"]},
    "contract": {"scope": "API contracts between services", "purpose": "producer and consumer agree on interface", "frameworks": ["Pact", "Spring Cloud Contract"]},
    "mutation": {"scope": "test quality", "how": "mutate code, check if tests catch it", "metric": "mutation score", "frameworks": ["Stryker", "mutmut", "pitest"]},
    "property-based": {"scope": "invariants over random inputs", "how": "generate random inputs, verify properties hold", "frameworks": {"python": "Hypothesis", "haskell": "QuickCheck", "javascript": "fast-check", "rust": "proptest"}},
    "snapshot": {"scope": "output stability", "how": "compare output to saved snapshot", "use": "UI components, serialized outputs", "frameworks": ["Jest snapshots", "insta (Rust)"]},
    "fuzz": {"scope": "crash/security bugs", "how": "random/semi-random input generation", "frameworks": ["AFL", "libFuzzer", "cargo-fuzz", "Atheris"]},
}

_TEST_PATTERNS = {
    "AAA": {"full": "Arrange-Act-Assert", "description": "Standard test structure: set up state, perform action, check result"},
    "given-when-then": {"description": "BDD-style: Given preconditions, When action, Then expected result", "alias": "GWT"},
    "test double": {"types": ["dummy (passed but unused)", "stub (returns fixed data)", "spy (records calls)", "mock (verifies interactions)", "fake (working simplified implementation)"]},
    "test pyramid": {"layers": ["unit (base, many)", "integration (middle)", "e2e (top, few)"], "principle": "more tests at the bottom, fewer at the top"},
    "test diamond": {"description": "Integration-heavy: thin unit layer, thick integration, thin e2e", "when": "CRUD apps, microservices"},
    "fixtures": {"description": "Reusable test setup/teardown", "frameworks": {"pytest": "@pytest.fixture", "junit": "@BeforeEach/@AfterEach"}},
    "parameterized": {"description": "Same test logic, multiple input/output pairs", "frameworks": {"pytest": "@pytest.mark.parametrize", "junit": "@ParameterizedTest"}},
    "golden file": {"description": "Compare output to saved 'golden' reference file", "alias": "approval testing"},
    "chaos engineering": {"description": "Intentionally inject failures to test resilience", "tools": ["Chaos Monkey", "Litmus", "Gremlin"]},
}

_COVERAGE_TYPES = {
    "line": {"description": "Percentage of lines executed", "typical_target": "80%+", "note": "Most common but weakest metric"},
    "branch": {"description": "Percentage of if/else branches taken", "typical_target": "70%+", "stronger_than": "line coverage"},
    "condition": {"description": "Each boolean sub-expression evaluated to true and false", "stronger_than": "branch"},
    "path": {"description": "All possible execution paths tested", "note": "exponential — impractical for complex code"},
    "mutation": {"description": "Percentage of code mutations caught by tests", "typical_target": "60%+", "note": "strongest practical metric"},
}


def test_type(name: str) -> dict:
    """Get details about a type of software testing."""
    key = str(name).lower().strip()
    entry = _TEST_TYPES.get(key)
    if not entry:
        for k, v in _TEST_TYPES.items():
            if key in k:
                return {"type": k, **v}
        return {"error": f"Unknown: {name}", "valid": sorted(_TEST_TYPES.keys())}
    return {"type": key, **entry}


def test_pattern(name: str) -> dict:
    """Get details about a testing pattern."""
    key = str(name).lower().strip()
    entry = _TEST_PATTERNS.get(key)
    if not entry:
        for k, v in _TEST_PATTERNS.items():
            if key in k.lower() or key == v.get("alias", "").lower():
                return {"pattern": k, **v}
        return {"error": f"Unknown: {name}", "valid": sorted(_TEST_PATTERNS.keys())}
    return {"pattern": key, **entry}


def coverage_type(name: str) -> dict:
    """Get details about a code coverage metric."""
    key = str(name).lower().strip()
    entry = _COVERAGE_TYPES.get(key)
    if not entry:
        for k, v in _COVERAGE_TYPES.items():
            if key in k:
                return {"type": k, **v}
        return {"error": f"Unknown: {name}", "valid": sorted(_COVERAGE_TYPES.keys())}
    return {"type": key, **entry}


def unit_vs_integration() -> dict:
    """Compare unit tests vs integration tests."""
    return {"unit": _TEST_TYPES["unit"], "integration": _TEST_TYPES["integration"],
            "summary": "Unit = isolated, fast, many. Integration = real deps, slower, fewer."}


def test_pyramid() -> dict:
    """Explain the test pyramid."""
    return _TEST_PATTERNS["test pyramid"]


def mock_vs_stub() -> dict:
    """Explain the difference between mocks and stubs."""
    return {
        "stub": "Returns fixed data. Tests STATE — did we get the right result?",
        "mock": "Verifies interactions. Tests BEHAVIOR — did we call the right things?",
        "spy": "Like a stub but also records what was called (hybrid).",
        "recommendation": "Prefer stubs for most tests. Use mocks sparingly — they couple tests to implementation.",
    }


TESTING_FUNCTIONS = {
    "test_type": test_type,
    "test_pattern": test_pattern,
    "coverage_type": coverage_type,
    "unit_vs_integration": unit_vs_integration,
    "test_pyramid": test_pyramid,
    "mock_vs_stub": mock_vs_stub,
}

TESTING_NL_PATTERNS = [
    (r'(?:what is|explain)\s+(?:a\s+)?(unit|integration|e2e|smoke|regression|performance|security|contract|mutation|property.based|snapshot|fuzz)\s+test', 'test_type("{0}")'),
    (r'(?:what is|explain)\s+(?:the\s+)?(AAA|arrange.act.assert|given.when.then|test pyramid|test diamond|fixtures?|parameterized|golden file|chaos)', 'test_pattern("{0}")'),
    (r'(?:compare|difference|vs)\s+unit\s+(?:and|vs)\s+integration', 'unit_vs_integration()'),
    (r'(?:compare|difference|vs)\s+mock\s+(?:and|vs)\s+stub', 'mock_vs_stub()'),
    (r'(?:what is|explain)\s+(line|branch|condition|path|mutation)\s+coverage', 'coverage_type("{0}")'),
]
