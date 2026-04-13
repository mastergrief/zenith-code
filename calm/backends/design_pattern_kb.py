"""
CALM Design Pattern knowledge backend — GoF patterns + modern patterns.

Models describe the wrong pattern by name. Stable facts.
"""

from __future__ import annotations

_DATA_VERSION = "2025-01"

# (category, intent, participants, when_to_use)
_PATTERNS = {
    "singleton": (
        "Creational", "Ensure a class has only one instance with a global access point",
        ["Singleton class", "static instance", "private constructor"],
        "Global state (config, logging, connection pool). Use sparingly — often a code smell.",
    ),
    "factory": (
        "Creational", "Create objects without specifying the exact class",
        ["Creator", "Product", "ConcreteCreator", "ConcreteProduct"],
        "When the exact type isn't known until runtime. Plugin systems, parsers.",
    ),
    "abstract factory": (
        "Creational", "Create families of related objects without specifying concrete classes",
        ["AbstractFactory", "ConcreteFactory", "AbstractProduct", "ConcreteProduct"],
        "Cross-platform UI, database drivers, theme systems.",
    ),
    "builder": (
        "Creational", "Construct complex objects step by step",
        ["Builder", "ConcreteBuilder", "Director", "Product"],
        "Objects with many optional parameters. Query builders, config objects.",
    ),
    "prototype": (
        "Creational", "Create new objects by cloning existing ones",
        ["Prototype", "ConcretePrototype", "Client"],
        "When construction is expensive. Object pools, undo systems.",
    ),
    "adapter": (
        "Structural", "Convert one interface to another that clients expect",
        ["Target", "Adapter", "Adaptee"],
        "Integrating third-party libraries. Legacy system wrappers.",
    ),
    "decorator": (
        "Structural", "Add responsibilities to objects dynamically",
        ["Component", "ConcreteComponent", "Decorator", "ConcreteDecorator"],
        "Extending behavior without subclassing. Middleware, logging, caching wrappers.",
    ),
    "observer": (
        "Behavioral", "Define a one-to-many dependency so dependents are notified of changes",
        ["Subject", "Observer", "ConcreteSubject", "ConcreteObserver"],
        "Event systems, pub/sub, reactive UI, webhooks.",
    ),
    "strategy": (
        "Behavioral", "Define a family of algorithms and make them interchangeable",
        ["Context", "Strategy", "ConcreteStrategy"],
        "Sorting algorithms, payment methods, compression strategies.",
    ),
    "command": (
        "Behavioral", "Encapsulate a request as an object",
        ["Command", "ConcreteCommand", "Invoker", "Receiver"],
        "Undo/redo, task queues, macro recording, transaction logging.",
    ),
    "iterator": (
        "Behavioral", "Access elements of a collection sequentially without exposing internals",
        ["Iterator", "ConcreteIterator", "Aggregate", "ConcreteAggregate"],
        "Custom collections, lazy evaluation, streaming data.",
    ),
    "state": (
        "Behavioral", "Allow object behavior to change when its state changes",
        ["Context", "State", "ConcreteState"],
        "Finite state machines, workflow engines, UI states.",
    ),
    "template method": (
        "Behavioral", "Define the skeleton of an algorithm, deferring steps to subclasses",
        ["AbstractClass", "ConcreteClass"],
        "Framework hooks, test fixtures, data processing pipelines.",
    ),
    "facade": (
        "Structural", "Provide a simplified interface to a complex subsystem",
        ["Facade", "Subsystem classes"],
        "API wrappers, library simplification, service aggregation.",
    ),
    "proxy": (
        "Structural", "Provide a surrogate or placeholder for another object",
        ["Proxy", "RealSubject", "Subject"],
        "Lazy loading, access control, caching, logging, remote objects.",
    ),
    "composite": (
        "Structural", "Compose objects into tree structures for part-whole hierarchies",
        ["Component", "Leaf", "Composite"],
        "File systems, UI component trees, organization charts.",
    ),
    "chain of responsibility": (
        "Behavioral", "Pass requests along a chain of handlers",
        ["Handler", "ConcreteHandler", "Client"],
        "Middleware pipelines, event bubbling, approval workflows.",
    ),
    "mediator": (
        "Behavioral", "Define an object that encapsulates how objects interact",
        ["Mediator", "ConcreteMediator", "Colleague"],
        "Chat rooms, air traffic control, form validation, MVC controllers.",
    ),
    # Modern patterns
    "repository": (
        "Modern", "Abstract data access behind a collection-like interface",
        ["Repository", "Entity", "DataMapper"],
        "Database abstraction, testability, domain-driven design.",
    ),
    "circuit breaker": (
        "Modern", "Prevent cascading failures by failing fast when a service is down",
        ["CircuitBreaker", "Service", "FallbackHandler"],
        "Microservice resilience, external API calls, database connections.",
    ),
    "saga": (
        "Modern", "Manage distributed transactions across multiple services",
        ["Saga", "SagaStep", "Compensator"],
        "Order processing, payment flows, multi-service operations.",
    ),
    "cqrs": (
        "Modern", "Separate read and write models for a data store",
        ["CommandHandler", "QueryHandler", "ReadModel", "WriteModel"],
        "High-read-to-write ratio, event sourcing, complex domains.",
    ),
}


def pattern_info(name: str) -> str:
    """Full info about a design pattern."""
    key = name.strip().lower()
    data = _PATTERNS.get(key)
    if not data:
        return f"unknown pattern: {name}"
    cat, intent, parts, when = data
    return f"{name} ({cat}): {intent}. Participants: {', '.join(parts)}. Use when: {when}"


def pattern_category(name: str) -> str:
    """Category of a design pattern (Creational/Structural/Behavioral/Modern)."""
    key = name.strip().lower()
    data = _PATTERNS.get(key)
    return data[0] if data else f"unknown: {name}"


def pattern_intent(name: str) -> str:
    """One-line intent of a design pattern."""
    key = name.strip().lower()
    data = _PATTERNS.get(key)
    return data[1] if data else f"unknown: {name}"


def pattern_participants(name: str) -> list:
    """Key participants/roles in a design pattern."""
    key = name.strip().lower()
    data = _PATTERNS.get(key)
    return data[2] if data else [f"unknown: {name}"]


def pattern_when(name: str) -> str:
    """When to use a design pattern."""
    key = name.strip().lower()
    data = _PATTERNS.get(key)
    return data[3] if data else f"unknown: {name}"


DESIGN_PATTERN_FUNCTIONS = {
    "pattern_info": pattern_info,
    "pattern_category": pattern_category,
    "pattern_intent": pattern_intent,
    "pattern_participants": pattern_participants,
    "pattern_when": pattern_when,
}

DESIGN_PATTERN_NL_PATTERNS = [
    (r'(?:what is|explain)\s+(?:the\s+)?(\w[\w\s]*?)\s+(?:design\s+)?pattern', 'pattern_info("{0}")'),
    (r'(?:when|why)\s+(?:should I|to)\s+use\s+(?:the\s+)?(\w[\w\s]*?)\s+pattern', 'pattern_when("{0}")'),
    (r'(?:participants?|roles?)\s+(?:in|of)\s+(?:the\s+)?(\w[\w\s]*?)\s+pattern', 'pattern_participants("{0}")'),
]
