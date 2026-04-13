"""
CALM HTTP backend — status codes, methods, headers, MIME types.

Models mix up 401 vs 403, confuse PUT vs PATCH, etc. Pure lookup tables.
"""

from __future__ import annotations

_STATUS_CODES = {
    100: "Continue", 101: "Switching Protocols",
    200: "OK", 201: "Created", 202: "Accepted", 204: "No Content",
    301: "Moved Permanently", 302: "Found", 304: "Not Modified",
    307: "Temporary Redirect", 308: "Permanent Redirect",
    400: "Bad Request", 401: "Unauthorized", 403: "Forbidden",
    404: "Not Found", 405: "Method Not Allowed", 408: "Request Timeout",
    409: "Conflict", 410: "Gone", 413: "Payload Too Large",
    415: "Unsupported Media Type", 418: "I'm a Teapot",
    422: "Unprocessable Entity", 429: "Too Many Requests",
    500: "Internal Server Error", 501: "Not Implemented",
    502: "Bad Gateway", 503: "Service Unavailable", 504: "Gateway Timeout",
}

_STATUS_CATEGORIES = {
    1: "Informational", 2: "Success", 3: "Redirection",
    4: "Client Error", 5: "Server Error",
}

_METHODS = {
    "GET": {"safe": True, "idempotent": True, "body": False},
    "HEAD": {"safe": True, "idempotent": True, "body": False},
    "POST": {"safe": False, "idempotent": False, "body": True},
    "PUT": {"safe": False, "idempotent": True, "body": True},
    "PATCH": {"safe": False, "idempotent": False, "body": True},
    "DELETE": {"safe": False, "idempotent": True, "body": False},
    "OPTIONS": {"safe": True, "idempotent": True, "body": False},
    "TRACE": {"safe": True, "idempotent": True, "body": False},
}

_MIME_TYPES = {
    "json": "application/json", "html": "text/html", "xml": "application/xml",
    "css": "text/css", "js": "application/javascript", "csv": "text/csv",
    "txt": "text/plain", "pdf": "application/pdf", "png": "image/png",
    "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif",
    "svg": "image/svg+xml", "webp": "image/webp", "mp4": "video/mp4",
    "mp3": "audio/mpeg", "zip": "application/zip", "gz": "application/gzip",
    "tar": "application/x-tar", "wasm": "application/wasm",
    "form": "application/x-www-form-urlencoded",
    "multipart": "multipart/form-data", "sse": "text/event-stream",
    "jsonl": "application/x-ndjson", "yaml": "application/yaml",
    "toml": "application/toml", "woff2": "font/woff2",
    "ico": "image/x-icon", "avif": "image/avif",
}


def http_status(code: int) -> str:
    """Look up HTTP status code meaning."""
    code = int(code)
    name = _STATUS_CODES.get(code)
    if name:
        cat = _STATUS_CATEGORIES.get(code // 100, "Unknown")
        return f"{code} {name} ({cat})"
    return f"{code} Unknown"


def http_status_category(code: int) -> str:
    """Classify status code: Informational/Success/Redirection/Client Error/Server Error."""
    return _STATUS_CATEGORIES.get(int(code) // 100, "Unknown")


def http_method_info(method: str) -> str:
    """Properties of an HTTP method: safe, idempotent, has body."""
    m = method.upper()
    info = _METHODS.get(m)
    if not info:
        return f"Unknown method: {m}"
    parts = []
    if info["safe"]:
        parts.append("safe")
    if info["idempotent"]:
        parts.append("idempotent")
    parts.append("has body" if info["body"] else "no body")
    return f"{m}: {', '.join(parts)}"


def http_is_safe(method: str) -> bool:
    """Whether an HTTP method is safe (no side effects)."""
    info = _METHODS.get(method.upper())
    return info["safe"] if info else False


def http_is_idempotent(method: str) -> bool:
    """Whether an HTTP method is idempotent."""
    info = _METHODS.get(method.upper())
    return info["idempotent"] if info else False


def mime_type(ext: str) -> str:
    """Look up MIME type for a file extension."""
    ext = ext.lower().lstrip(".")
    return _MIME_TYPES.get(ext, f"unknown extension: {ext}")


def mime_category(ext: str) -> str:
    """Category of a MIME type: text, image, audio, video, application, etc."""
    mt = _MIME_TYPES.get(ext.lower().lstrip("."), "")
    return mt.split("/")[0] if "/" in mt else "unknown"


HTTP_FUNCTIONS = {
    "http_status": http_status,
    "http_status_category": http_status_category,
    "http_method_info": http_method_info,
    "http_is_safe": http_is_safe,
    "http_is_idempotent": http_is_idempotent,
    "mime_type": mime_type,
    "mime_category": mime_category,
}
