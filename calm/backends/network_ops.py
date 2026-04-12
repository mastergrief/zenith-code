"""
CALM network backend — verified URL/HTTP/DNS operations.

The model writes "this URL has 3 query params" — the engine parses
and counts.

Functions: URL parsing, HTTP status codes, IP validation, CIDR check.
"""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse


def url_parse(url: str) -> dict:
    """Parse a URL into components."""
    p = urlparse(url)
    params = parse_qs(p.query)
    return {
        "scheme": p.scheme,
        "host": p.hostname or "",
        "port": p.port,
        "path": p.path,
        "query": dict(params),
        "query_count": len(params),
        "fragment": p.fragment,
        "valid": bool(p.scheme and p.hostname),
    }


def url_build(scheme: str, host: str, path: str = "/",
              query: dict = None, port: int = None) -> str:
    """Build a URL from components."""
    netloc = host
    if port:
        netloc = f"{host}:{port}"
    q = urlencode(query or {}, doseq=True)
    return urlunparse((scheme, netloc, path, "", q, ""))


def http_status(code: int) -> dict:
    """Look up HTTP status code meaning."""
    code = int(code)
    codes = {
        200: "OK", 201: "Created", 204: "No Content",
        301: "Moved Permanently", 302: "Found", 304: "Not Modified",
        400: "Bad Request", 401: "Unauthorized", 403: "Forbidden",
        404: "Not Found", 405: "Method Not Allowed", 409: "Conflict",
        413: "Payload Too Large", 422: "Unprocessable Entity",
        429: "Too Many Requests",
        500: "Internal Server Error", 502: "Bad Gateway",
        503: "Service Unavailable", 504: "Gateway Timeout",
    }
    category = (
        "informational" if 100 <= code < 200 else
        "success" if 200 <= code < 300 else
        "redirect" if 300 <= code < 400 else
        "client_error" if 400 <= code < 500 else
        "server_error" if 500 <= code < 600 else
        "unknown"
    )
    return {
        "code": code,
        "meaning": codes.get(code, "Unknown"),
        "category": category,
    }


def is_valid_ip(addr: str) -> bool:
    """Check if a string is a valid IPv4 or IPv6 address."""
    try:
        ipaddress.ip_address(addr)
        return True
    except ValueError:
        return False


def ip_info(addr: str) -> dict:
    """Get info about an IP address."""
    try:
        ip = ipaddress.ip_address(addr)
        return {
            "address": str(ip),
            "version": ip.version,
            "is_private": ip.is_private,
            "is_loopback": ip.is_loopback,
            "is_link_local": ip.is_link_local,
            "is_multicast": ip.is_multicast,
        }
    except ValueError:
        return {"error": f"invalid IP: {addr}"}


def cidr_contains(network: str, addr: str) -> bool:
    """Check if an IP address is within a CIDR range."""
    try:
        net = ipaddress.ip_network(network, strict=False)
        ip = ipaddress.ip_address(addr)
        return ip in net
    except ValueError:
        return False


def cidr_info(network: str) -> dict:
    """Get info about a CIDR network."""
    try:
        net = ipaddress.ip_network(network, strict=False)
        return {
            "network": str(net.network_address),
            "broadcast": str(net.broadcast_address),
            "netmask": str(net.netmask),
            "prefix_len": net.prefixlen,
            "num_addresses": net.num_addresses,
            "is_private": net.is_private,
        }
    except ValueError as e:
        return {"error": str(e)}


def is_valid_email(email: str) -> bool:
    """Basic email validation via regex."""
    return bool(re.match(
        r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$', email
    ))


def is_valid_domain(domain: str) -> bool:
    """Basic domain name validation."""
    return bool(re.match(
        r'^(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$',
        domain
    ))


NETWORK_FUNCTIONS = {
    "url_parse": url_parse,
    "url_build": url_build,
    "http_status": http_status,
    "is_valid_ip": is_valid_ip,
    "ip_info": ip_info,
    "cidr_contains": cidr_contains,
    "cidr_info": cidr_info,
    "is_valid_email": is_valid_email,
    "is_valid_domain": is_valid_domain,
}
