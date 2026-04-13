"""
CALM Cloud/distributed patterns knowledge backend — microservices, patterns, fallacies.

Models confuse patterns, mix up CAP implications, hallucinate architecture details.
"""

from __future__ import annotations

_DATA_VERSION = "2025-01"

_PATTERNS = {
    "circuit breaker": {"description": "Stop calling a failing service. After N failures, open circuit → return fallback. Periodically try again (half-open).", "states": ["closed (normal)", "open (failing)", "half-open (testing)"], "tools": ["Hystrix", "resilience4j", "Polly"], "use": "external API calls, database connections"},
    "retry": {"description": "Automatically retry failed operations with backoff", "strategies": ["fixed delay", "exponential backoff", "exponential + jitter"], "gotcha": "only retry idempotent operations; non-idempotent → at-most-once or exactly-once", "jitter_why": "prevents thundering herd on retries"},
    "bulkhead": {"description": "Isolate components so failure in one doesn't cascade. Like ship compartments.", "implementations": ["thread pool isolation", "process isolation", "separate service instances"], "use": "prevent one slow dependency from consuming all resources"},
    "saga": {"description": "Manage distributed transactions without 2PC. Each step has a compensating action.", "types": ["choreography (events)", "orchestration (coordinator)"], "compensating": "if step 3 fails, run undo for steps 2, 1", "use": "multi-service business transactions"},
    "CQRS": {"full": "Command Query Responsibility Segregation", "description": "Separate read and write models. Different DBs/schemas for queries vs commands.", "use": "complex domains, different read/write scaling needs", "often_paired_with": "event sourcing"},
    "event sourcing": {"description": "Store all state changes as immutable events. Current state = replay of events.", "pros": ["full audit trail", "temporal queries", "event replay"], "cons": ["eventual consistency", "complexity", "storage growth"], "use": "financial systems, audit-critical domains"},
    "sidecar": {"description": "Attach helper process alongside main service (same pod/host). Handles cross-cutting concerns.", "examples": ["Envoy proxy", "log collector", "config agent"], "use": "service mesh, logging, monitoring without modifying app code"},
    "strangler fig": {"description": "Gradually replace legacy system by routing requests to new system. Old and new run in parallel.", "named_after": "fig vine that gradually replaces its host tree", "use": "legacy migration without big-bang rewrite"},
    "ambassador": {"description": "Proxy that handles connections on behalf of the service (retry, circuit breaking, TLS)", "vs_sidecar": "ambassador = network proxy; sidecar = general helper"},
    "backends for frontends": {"alias": "BFF", "description": "Separate backend per frontend type (mobile BFF, web BFF, IoT BFF)", "use": "different data needs per client, API aggregation"},
    "event-driven": {"description": "Services communicate via events (async, decoupled). Producer doesn't know consumers.", "implementations": ["Kafka", "RabbitMQ", "AWS EventBridge", "NATS"], "patterns": ["pub/sub", "event streaming", "event notification"]},
    "service mesh": {"description": "Infrastructure layer handling service-to-service communication (mTLS, routing, observability)", "implementations": ["Istio", "Linkerd", "Consul Connect"], "provides": ["traffic management", "security (mTLS)", "observability"]},
}

_FALLACIES = {
    "1": {"fallacy": "The network is reliable", "truth": "Packets drop, connections fail, DNS breaks. Design for it."},
    "2": {"fallacy": "Latency is zero", "truth": "Every network call adds ms-hundreds of ms. Minimize round trips."},
    "3": {"fallacy": "Bandwidth is infinite", "truth": "Bandwidth costs money and has limits. Don't send unnecessary data."},
    "4": {"fallacy": "The network is secure", "truth": "Assume the network is hostile. Encrypt everything (mTLS, TLS)."},
    "5": {"fallacy": "Topology doesn't change", "truth": "Services move, IPs change, regions fail. Use service discovery."},
    "6": {"fallacy": "There is one administrator", "truth": "Multiple teams, orgs, cloud providers. Coordinate carefully."},
    "7": {"fallacy": "Transport cost is zero", "truth": "Serialization, network I/O, load balancers all have costs."},
    "8": {"fallacy": "The network is homogeneous", "truth": "Different protocols, versions, hardware. Design for heterogeneity."},
}

_TWELVE_FACTOR = {
    "1": {"name": "Codebase", "rule": "One codebase tracked in VCS, many deploys"},
    "2": {"name": "Dependencies", "rule": "Explicitly declare and isolate dependencies"},
    "3": {"name": "Config", "rule": "Store config in the environment (not code)"},
    "4": {"name": "Backing services", "rule": "Treat backing services as attached resources"},
    "5": {"name": "Build, release, run", "rule": "Strictly separate build, release, run stages"},
    "6": {"name": "Processes", "rule": "Execute the app as one or more stateless processes"},
    "7": {"name": "Port binding", "rule": "Export services via port binding"},
    "8": {"name": "Concurrency", "rule": "Scale out via the process model"},
    "9": {"name": "Disposability", "rule": "Maximize robustness with fast startup and graceful shutdown"},
    "10": {"name": "Dev/prod parity", "rule": "Keep development, staging, production as similar as possible"},
    "11": {"name": "Logs", "rule": "Treat logs as event streams"},
    "12": {"name": "Admin processes", "rule": "Run admin/management tasks as one-off processes"},
}


def cloud_pattern(name: str) -> dict:
    """Get details about a cloud/distributed systems pattern."""
    key = str(name).lower().strip()
    for k, v in _PATTERNS.items():
        if key in k.lower() or k.lower() in key:
            return {"pattern": k, **v}
    return {"error": f"Unknown: {name}", "valid": sorted(_PATTERNS.keys())}


def fallacy(number: int) -> dict:
    """Get a fallacy of distributed computing (1-8)."""
    entry = _FALLACIES.get(str(int(number)))
    if not entry:
        return {"error": f"Unknown: {number} (valid: 1-8)"}
    return {"number": int(number), **entry}


def all_fallacies() -> list[dict]:
    """List all 8 fallacies of distributed computing."""
    return [{"number": int(k), **v} for k, v in sorted(_FALLACIES.items())]


def twelve_factor(number: int) -> dict:
    """Get a twelve-factor app principle (1-12)."""
    entry = _TWELVE_FACTOR.get(str(int(number)))
    if not entry:
        return {"error": f"Unknown: {number} (valid: 1-12)"}
    return {"number": int(number), **entry}


def all_twelve_factors() -> list[dict]:
    """List all 12 factors."""
    return [{"number": int(k), **v} for k, v in sorted(_TWELVE_FACTOR.items(), key=lambda x: int(x[0]))]


def saga_vs_2pc() -> dict:
    """Compare Saga pattern vs Two-Phase Commit."""
    return {
        "saga": {"type": "eventual consistency", "coordination": "choreography or orchestration", "isolation": "no — requires compensating transactions", "availability": "high", "complexity": "high (compensation logic)"},
        "2PC": {"type": "strong consistency", "coordination": "coordinator (transaction manager)", "isolation": "yes (distributed lock)", "availability": "low (coordinator is SPOF)", "complexity": "moderate but brittle"},
        "recommendation": "Use sagas for microservices. 2PC only within a single database system.",
    }


def circuit_breaker_states() -> dict:
    """Explain circuit breaker states."""
    return {
        "closed": "Normal operation. Requests pass through. Failures counted.",
        "open": "Too many failures. All requests fail fast (return fallback). Timer starts.",
        "half_open": "Timer expired. Allow one test request. Success → closed. Failure → open again.",
    }


CLOUD_PATTERNS_FUNCTIONS = {
    "cloud_pattern": cloud_pattern,
    "fallacy": fallacy,
    "all_fallacies": all_fallacies,
    "twelve_factor": twelve_factor,
    "all_twelve_factors": all_twelve_factors,
    "saga_vs_2pc": saga_vs_2pc,
    "circuit_breaker_states": circuit_breaker_states,
}

CLOUD_PATTERNS_NL_PATTERNS = [
    (r'(?:what is|explain)\s+(?:the\s+)?(circuit breaker|retry|bulkhead|saga|CQRS|event sourcing|sidecar|strangler|ambassador|BFF|event.driven|service mesh)\s+pattern', 'cloud_pattern("{0}")'),
    (r'(?:what is|explain)\s+(?:the\s+)?(\d)(?:st|nd|rd|th)\s+fallacy', 'fallacy({0})'),
    (r'(?:list|what are)\s+(?:the\s+)?(?:8\s+)?fallacies\s+(?:of\s+)?distributed', 'all_fallacies()'),
    (r'(?:what is|explain)\s+(?:the\s+)?(\d+)(?:st|nd|rd|th)\s+(?:twelve.factor|12.factor)', 'twelve_factor({0})'),
    (r'(?:list|what are)\s+(?:the\s+)?(?:12|twelve)\s+factors?', 'all_twelve_factors()'),
    (r'(?:compare|difference|vs)\s+saga\s+(?:and|vs)\s+2PC', 'saga_vs_2pc()'),
    (r'(?:circuit breaker)\s+states', 'circuit_breaker_states()'),
]
