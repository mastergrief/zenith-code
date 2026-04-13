"""
CALM HTTP reference knowledge backend — headers, CORS, caching, content types.

Models confuse Cache-Control directives, botch CORS headers, hallucinate content types.
"""

from __future__ import annotations

_DATA_VERSION = "2025-01"

_CACHE_DIRECTIVES = {
    "no-store": {"scope": "request+response", "effect": "Never cache. Nothing stored.", "use": "sensitive data (banking, auth)"},
    "no-cache": {"scope": "request+response", "effect": "Cache but revalidate before use", "use": "always-fresh content", "gotcha": "Does NOT mean 'don't cache' — it means 'always revalidate'"},
    "public": {"scope": "response", "effect": "Any cache (CDN, proxy, browser) can store", "use": "static assets, public pages"},
    "private": {"scope": "response", "effect": "Only browser cache, not shared caches", "use": "user-specific content"},
    "max-age": {"scope": "request+response", "effect": "Response is fresh for N seconds", "use": "static assets (e.g. max-age=31536000 for 1 year)"},
    "s-maxage": {"scope": "response", "effect": "Like max-age but only for shared caches (CDN)", "use": "CDN caching separate from browser"},
    "must-revalidate": {"scope": "response", "effect": "Once stale, MUST revalidate (no stale-while-revalidate)", "use": "critical data that must be current"},
    "stale-while-revalidate": {"scope": "response", "effect": "Serve stale while revalidating in background", "use": "progressive loading patterns"},
    "immutable": {"scope": "response", "effect": "Content will never change (skip conditional requests)", "use": "versioned/fingerprinted assets"},
}

_CORS_HEADERS = {
    "Access-Control-Allow-Origin": {"type": "response", "values": "origin URL or *", "preflight": True, "gotcha": "* cannot be used with credentials"},
    "Access-Control-Allow-Methods": {"type": "response (preflight)", "values": "GET, POST, PUT, DELETE, etc.", "preflight": True},
    "Access-Control-Allow-Headers": {"type": "response (preflight)", "values": "Content-Type, Authorization, etc.", "preflight": True},
    "Access-Control-Allow-Credentials": {"type": "response", "values": "true", "gotcha": "Origin must be specific (not *) when true"},
    "Access-Control-Max-Age": {"type": "response (preflight)", "values": "seconds", "effect": "Cache preflight response"},
    "Access-Control-Expose-Headers": {"type": "response", "values": "header names", "effect": "Makes non-simple headers readable by JS"},
    "Origin": {"type": "request", "values": "requesting origin URL", "auto_sent": True},
}

_SECURITY_HEADERS = {
    "Content-Security-Policy": {"purpose": "Prevent XSS, injection", "example": "default-src 'self'; script-src 'self'", "critical": True},
    "Strict-Transport-Security": {"purpose": "Force HTTPS", "example": "max-age=31536000; includeSubDomains", "alias": "HSTS", "critical": True},
    "X-Content-Type-Options": {"purpose": "Prevent MIME sniffing", "example": "nosniff", "critical": True},
    "X-Frame-Options": {"purpose": "Prevent clickjacking", "example": "DENY or SAMEORIGIN", "deprecated_by": "CSP frame-ancestors"},
    "Referrer-Policy": {"purpose": "Control Referer header leakage", "example": "strict-origin-when-cross-origin"},
    "Permissions-Policy": {"purpose": "Control browser features (camera, mic, geolocation)", "example": "camera=(), microphone=()"},
    "X-XSS-Protection": {"purpose": "Legacy XSS filter", "example": "0", "note": "Set to 0 — CSP is better; the filter can introduce vulnerabilities"},
}

_CONTENT_TYPES = {
    "text/html": {"extension": ".html", "charset": "UTF-8 recommended", "category": "text"},
    "text/plain": {"extension": ".txt", "charset": "UTF-8", "category": "text"},
    "text/css": {"extension": ".css", "charset": "UTF-8", "category": "text"},
    "text/csv": {"extension": ".csv", "charset": "UTF-8", "category": "text"},
    "application/json": {"extension": ".json", "charset": "UTF-8", "category": "data"},
    "application/xml": {"extension": ".xml", "charset": "UTF-8", "category": "data"},
    "application/javascript": {"extension": ".js", "charset": "UTF-8", "category": "script", "note": "text/javascript also valid"},
    "application/pdf": {"extension": ".pdf", "category": "document"},
    "application/zip": {"extension": ".zip", "category": "archive"},
    "application/gzip": {"extension": ".gz", "category": "archive"},
    "application/octet-stream": {"extension": "*", "category": "binary", "use": "unknown binary data, force download"},
    "application/x-www-form-urlencoded": {"extension": None, "category": "form", "use": "HTML form submission (default)"},
    "multipart/form-data": {"extension": None, "category": "form", "use": "file upload, form with binary data"},
    "image/png": {"extension": ".png", "category": "image"},
    "image/jpeg": {"extension": ".jpg/.jpeg", "category": "image"},
    "image/gif": {"extension": ".gif", "category": "image"},
    "image/svg+xml": {"extension": ".svg", "category": "image"},
    "image/webp": {"extension": ".webp", "category": "image"},
    "audio/mpeg": {"extension": ".mp3", "category": "audio"},
    "video/mp4": {"extension": ".mp4", "category": "video"},
    "font/woff2": {"extension": ".woff2", "category": "font"},
}


def cache_directive(name: str) -> dict:
    """Get details about an HTTP Cache-Control directive."""
    key = str(name).lower().strip().replace("_", "-")
    entry = _CACHE_DIRECTIVES.get(key)
    if not entry:
        return {"error": f"Unknown directive: {name}", "valid": list(_CACHE_DIRECTIVES.keys())}
    return {"directive": key, **entry}


def cors_header(name: str) -> dict:
    """Get details about a CORS header."""
    key = str(name).strip()
    # Try exact match first
    entry = _CORS_HEADERS.get(key)
    if not entry:
        # Case-insensitive
        for k, v in _CORS_HEADERS.items():
            if k.lower() == key.lower():
                return {"header": k, **v}
        return {"error": f"Unknown header: {name}", "valid": list(_CORS_HEADERS.keys())}
    return {"header": key, **entry}


def security_header(name: str) -> dict:
    """Get details about a security header."""
    key = str(name).strip()
    entry = _SECURITY_HEADERS.get(key)
    if not entry:
        for k, v in _SECURITY_HEADERS.items():
            if k.lower() == key.lower() or v.get("alias", "").lower() == key.lower():
                return {"header": k, **v}
        return {"error": f"Unknown header: {name}", "valid": list(_SECURITY_HEADERS.keys())}
    return {"header": key, **entry}


def content_type(mime: str) -> dict:
    """Get info about a MIME/content type."""
    key = str(mime).lower().strip()
    entry = _CONTENT_TYPES.get(key)
    if not entry:
        # Try by extension
        for k, v in _CONTENT_TYPES.items():
            ext = v.get("extension", "")
            if ext and key.lstrip(".") in str(ext):
                return {"mime": k, **v}
        return {"error": f"Unknown type: {mime}", "valid": sorted(_CONTENT_TYPES.keys())}
    return {"mime": key, **entry}


def list_security_headers() -> list[str]:
    """List all recommended security headers."""
    return list(_SECURITY_HEADERS.keys())


def no_cache_vs_no_store() -> dict:
    """Explain the difference between no-cache and no-store."""
    return {
        "no-cache": "Cache the response, but revalidate with server before every use (conditional GET). Content IS stored.",
        "no-store": "Never store the response anywhere. No caching at all.",
        "common_mistake": "'no-cache' does NOT mean 'don't cache'. It means 'always ask the server if it's still valid'.",
        "for_sensitive_data": "Use no-store, not no-cache",
    }


HTTP_REF_FUNCTIONS = {
    "cache_directive": cache_directive,
    "cors_header": cors_header,
    "security_header": security_header,
    "content_type": content_type,
    "list_security_headers": list_security_headers,
    "no_cache_vs_no_store": no_cache_vs_no_store,
}

HTTP_REF_NL_PATTERNS = [
    (r'(?:what is|explain)\s+(?:cache.control\s+)?(no.cache|no.store|public|private|max.age|immutable|must.revalidate|stale.while.revalidate)', 'cache_directive("{0}")'),
    (r'(?:difference between|vs)\s+no.cache\s+(?:and|vs)\s+no.store', 'no_cache_vs_no_store()'),
    (r'(?:what is|explain)\s+(?:the\s+)?(CORS|cors)\s+header\s+(\S+)', 'cors_header("{1}")'),
    (r'(?:what is|explain)\s+(Content-Security-Policy|HSTS|CSP|X-Frame-Options|Referrer-Policy)', 'security_header("{0}")'),
    (r'(?:content type|MIME type)\s+(?:for|of)\s+(?:a\s+)?(\.\w+|\w+/\w+)', 'content_type("{0}")'),
    (r'(?:what are|list)\s+(?:recommended\s+)?security headers', 'list_security_headers()'),
]
