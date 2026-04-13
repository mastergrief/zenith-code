"""
CALM API design patterns knowledge backend — REST, GraphQL, gRPC, pagination.

Models mix up REST principles, confuse status codes for operations, hallucinate API patterns.
"""

from __future__ import annotations

_DATA_VERSION = "2025-01"

_REST_PRINCIPLES = {
    "uniform interface": {"description": "Consistent resource-based URLs, standard HTTP methods, self-descriptive messages", "constraints": ["resource identification (URLs)", "manipulation through representations", "self-descriptive messages", "HATEOAS"]},
    "stateless": {"description": "Each request contains all information needed. Server stores no client state.", "benefit": "scalability, reliability", "exception": "auth tokens are metadata, not state"},
    "client-server": {"description": "Separation of concerns. Client handles UI, server handles data/logic."},
    "cacheable": {"description": "Responses must declare themselves cacheable or not (Cache-Control, ETag, Last-Modified)"},
    "layered system": {"description": "Client can't tell if connected directly to server or through proxy/load balancer"},
    "code on demand": {"description": "Server can send executable code to client (optional)", "example": "JavaScript in web responses"},
}

_HTTP_METHODS_REST = {
    "GET": {"operation": "Read", "idempotent": True, "safe": True, "body": "no", "response": "resource", "status": "200 OK"},
    "POST": {"operation": "Create", "idempotent": False, "safe": False, "body": "yes", "response": "created resource", "status": "201 Created"},
    "PUT": {"operation": "Replace (full update)", "idempotent": True, "safe": False, "body": "yes (full resource)", "response": "updated resource", "status": "200 OK"},
    "PATCH": {"operation": "Partial update", "idempotent": False, "safe": False, "body": "yes (partial)", "response": "updated resource", "status": "200 OK"},
    "DELETE": {"operation": "Delete", "idempotent": True, "safe": False, "body": "optional", "response": "empty or confirmation", "status": "204 No Content"},
    "HEAD": {"operation": "Same as GET but no body", "idempotent": True, "safe": True, "body": "no", "response": "headers only"},
    "OPTIONS": {"operation": "Show allowed methods", "idempotent": True, "safe": True, "body": "no", "response": "Allow header", "use": "CORS preflight"},
}

_PAGINATION = {
    "offset": {"params": "?offset=20&limit=10", "pros": ["simple", "jump to any page"], "cons": ["slow on large offsets (O(n))", "items shift when data changes"], "sql": "LIMIT 10 OFFSET 20"},
    "cursor": {"params": "?cursor=abc123&limit=10", "pros": ["fast (O(1))", "stable during mutations"], "cons": ["can't jump to arbitrary page", "opaque cursor"], "used_by": ["Slack", "Stripe", "GitHub GraphQL"]},
    "keyset": {"params": "?after_id=100&limit=10", "pros": ["fast (uses index)", "stable"], "cons": ["requires sortable unique key"], "sql": "WHERE id > 100 ORDER BY id LIMIT 10", "alias": "seek pagination"},
    "page": {"params": "?page=3&per_page=10", "pros": ["user-friendly"], "cons": ["same as offset internally"], "used_by": ["GitHub REST", "most simple APIs"]},
}

_API_STYLES = {
    "REST": {"protocol": "HTTP", "data_format": "JSON (usually)", "contract": "implicit (OpenAPI for docs)", "versioning": ["URL (/v1/)", "header (Accept: v1)", "query (?version=1)"], "strengths": ["simple", "cacheable", "standard tooling"], "weaknesses": ["over-fetching", "multiple round trips"]},
    "GraphQL": {"protocol": "HTTP (single endpoint)", "data_format": "JSON", "contract": "schema (SDL)", "versioning": "field deprecation (no versions)", "strengths": ["client specifies fields", "single request for related data", "introspection"], "weaknesses": ["N+1 queries on server", "caching harder", "learning curve"]},
    "gRPC": {"protocol": "HTTP/2", "data_format": "Protocol Buffers (binary)", "contract": ".proto files (strict)", "versioning": "proto field numbers (backward compatible)", "strengths": ["fast (binary)", "streaming", "code generation", "strongly typed"], "weaknesses": ["not browser-native", "harder to debug", "requires proto toolchain"]},
    "WebSocket": {"protocol": "WebSocket (ws/wss)", "data_format": "any", "contract": "none (custom)", "strengths": ["full-duplex", "real-time", "low latency"], "weaknesses": ["stateful", "scaling harder", "no standard request/response"]},
    "webhook": {"protocol": "HTTP callback", "data_format": "JSON (usually)", "direction": "server→client (push)", "strengths": ["real-time notifications", "no polling"], "weaknesses": ["client must expose endpoint", "retry/delivery guarantees"]},
}

_AUTH_PATTERNS = {
    "API key in header": {"header": "X-API-Key: <key>", "security": "moderate", "use": "simple APIs, rate limiting"},
    "Bearer token": {"header": "Authorization: Bearer <token>", "security": "good", "use": "OAuth2/JWT-based APIs", "standard": "RFC 6750"},
    "Basic auth": {"header": "Authorization: Basic <base64(user:pass)>", "security": "weak (base64 is NOT encryption)", "use": "internal APIs only, MUST use HTTPS"},
    "OAuth2 PKCE": {"flow": "authorization code + code verifier", "security": "strong", "use": "SPAs, mobile apps, public clients"},
    "mutual TLS": {"mechanism": "client certificate in TLS handshake", "security": "very strong", "use": "service-to-service, zero-trust"},
}


def rest_principle(name: str) -> dict:
    """Get a REST architectural principle."""
    key = str(name).lower().strip()
    for k, v in _REST_PRINCIPLES.items():
        if key in k:
            return {"principle": k, **v}
    return {"error": f"Unknown: {name}", "valid": list(_REST_PRINCIPLES.keys())}


def http_method_rest(method: str) -> dict:
    """Get REST semantics for an HTTP method."""
    key = str(method).upper().strip()
    entry = _HTTP_METHODS_REST.get(key)
    if not entry:
        return {"error": f"Unknown: {method}", "valid": list(_HTTP_METHODS_REST.keys())}
    return {"method": key, **entry}


def pagination_style(style: str) -> dict:
    """Get details about an API pagination style."""
    key = str(style).lower().strip()
    entry = _PAGINATION.get(key)
    if not entry:
        for k, v in _PAGINATION.items():
            if key in k or key == v.get("alias", ""):
                return {"style": k, **v}
        return {"error": f"Unknown: {style}", "valid": list(_PAGINATION.keys())}
    return {"style": key, **entry}


def api_style(name: str) -> dict:
    """Get details about an API architectural style."""
    key = str(name).strip()
    for k, v in _API_STYLES.items():
        if key.lower() in k.lower():
            return {"style": k, **v}
    return {"error": f"Unknown: {name}", "valid": list(_API_STYLES.keys())}


def rest_vs_graphql() -> dict:
    """Compare REST and GraphQL."""
    return {"REST": _API_STYLES["REST"], "GraphQL": _API_STYLES["GraphQL"]}


def rest_vs_grpc() -> dict:
    """Compare REST and gRPC."""
    return {"REST": _API_STYLES["REST"], "gRPC": _API_STYLES["gRPC"]}


def put_vs_patch() -> dict:
    """Explain the difference between PUT and PATCH."""
    return {"PUT": _HTTP_METHODS_REST["PUT"], "PATCH": _HTTP_METHODS_REST["PATCH"],
            "summary": "PUT replaces the entire resource. PATCH updates only specified fields."}


def cursor_vs_offset() -> dict:
    """Compare cursor and offset pagination."""
    return {"cursor": _PAGINATION["cursor"], "offset": _PAGINATION["offset"]}


def idempotent_methods() -> list[str]:
    """List all idempotent HTTP methods."""
    return [k for k, v in _HTTP_METHODS_REST.items() if v.get("idempotent")]


API_PATTERNS_FUNCTIONS = {
    "rest_principle": rest_principle,
    "http_method_rest": http_method_rest,
    "pagination_style": pagination_style,
    "api_style": api_style,
    "rest_vs_graphql": rest_vs_graphql,
    "rest_vs_grpc": rest_vs_grpc,
    "put_vs_patch": put_vs_patch,
    "cursor_vs_offset": cursor_vs_offset,
    "idempotent_methods": idempotent_methods,
}

API_PATTERNS_NL_PATTERNS = [
    (r'(?:what is|explain)\s+(?:the\s+)?REST\s+(?:principle\s+)?(stateless|uniform|cacheable|layered|client.server)', 'rest_principle("{0}")'),
    (r'(?:compare|difference|vs)\s+REST\s+(?:and|vs)\s+GraphQL', 'rest_vs_graphql()'),
    (r'(?:compare|difference|vs)\s+REST\s+(?:and|vs)\s+gRPC', 'rest_vs_grpc()'),
    (r'(?:compare|difference|vs)\s+PUT\s+(?:and|vs)\s+PATCH', 'put_vs_patch()'),
    (r'(?:compare|difference|vs)\s+cursor\s+(?:and|vs)\s+offset\s+pagination', 'cursor_vs_offset()'),
    (r'(?:what is|explain)\s+(cursor|offset|keyset|page)\s+pagination', 'pagination_style("{0}")'),
    (r'(?:which|what)\s+(?:HTTP\s+)?methods?\s+(?:are\s+)?idempotent', 'idempotent_methods()'),
    (r'(?:what is|explain)\s+(REST|GraphQL|gRPC|WebSocket|webhook)', 'api_style("{0}")'),
]
