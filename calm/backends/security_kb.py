"""
CALM Security knowledge backend — OWASP, vulnerability types, auth methods.

Models confuse vulnerability types, mix up auth mechanisms, hallucinate mitigations.
"""

from __future__ import annotations

_DATA_VERSION = "2025-01"

_OWASP_TOP_10 = {
    "A01": {"name": "Broken Access Control", "description": "Users can act outside intended permissions", "examples": ["IDOR", "privilege escalation", "missing function-level access control"], "mitigations": ["deny by default", "RBAC/ABAC", "server-side enforcement"]},
    "A02": {"name": "Cryptographic Failures", "description": "Sensitive data exposure through weak crypto", "examples": ["plaintext passwords", "weak algorithms (MD5, SHA1)", "missing TLS"], "mitigations": ["encrypt at rest + transit", "use strong algorithms (AES-256, bcrypt)", "HSTS"]},
    "A03": {"name": "Injection", "description": "Untrusted data sent to interpreter as command/query", "examples": ["SQL injection", "XSS", "command injection", "LDAP injection"], "mitigations": ["parameterized queries", "input validation", "ORM", "CSP"]},
    "A04": {"name": "Insecure Design", "description": "Missing or ineffective security controls by design", "examples": ["no rate limiting", "predictable tokens", "missing threat modeling"], "mitigations": ["threat modeling", "secure design patterns", "abuse case testing"]},
    "A05": {"name": "Security Misconfiguration", "description": "Insecure defaults, open cloud storage, verbose errors", "examples": ["default credentials", "directory listing", "stack traces in production"], "mitigations": ["hardened defaults", "remove unused features", "automated config scanning"]},
    "A06": {"name": "Vulnerable Components", "description": "Using components with known vulnerabilities", "examples": ["outdated libraries", "unpatched OS", "CVEs in dependencies"], "mitigations": ["SCA tools (Snyk, Dependabot)", "version pinning", "regular updates"]},
    "A07": {"name": "Authentication Failures", "description": "Broken authentication and session management", "examples": ["credential stuffing", "weak passwords", "missing MFA", "session fixation"], "mitigations": ["MFA", "rate limiting", "secure session management", "argon2 hashing"]},
    "A08": {"name": "Software and Data Integrity", "description": "Code/data integrity not verified", "examples": ["CI/CD pipeline compromise", "unsigned updates", "deserialization attacks"], "mitigations": ["digital signatures", "SBOM", "integrity checks"]},
    "A09": {"name": "Logging and Monitoring Failures", "description": "Insufficient logging, monitoring, alerting", "examples": ["no audit logs", "no alerting on brute force", "logs not centralized"], "mitigations": ["centralized logging", "SIEM", "alerting rules", "tamper-proof logs"]},
    "A10": {"name": "SSRF", "description": "Server-Side Request Forgery — server makes requests to attacker-controlled URLs", "examples": ["cloud metadata access (169.254.169.254)", "internal port scanning", "protocol smuggling"], "mitigations": ["allowlist URLs", "deny internal networks", "disable redirects"]},
}

_AUTH_METHODS = {
    "session": {"type": "stateful", "storage": "server-side", "transport": "cookie", "pros": ["simple", "revocable"], "cons": ["server memory", "not REST-friendly", "CSRF risk"]},
    "JWT": {"type": "stateless", "storage": "client-side", "transport": "Authorization header", "pros": ["scalable", "no server state", "cross-domain"], "cons": ["can't revoke easily", "payload visible (base64)", "size"]},
    "OAuth2": {"type": "delegation", "flow": ["authorization code", "PKCE", "client credentials", "device code"], "use": "third-party access (Login with Google)", "NOT_for": "authentication (use OIDC on top)"},
    "OIDC": {"type": "identity", "built_on": "OAuth2", "adds": "ID token (JWT with user info)", "use": "authentication + identity"},
    "API key": {"type": "static token", "transport": "header or query param", "pros": ["simple"], "cons": ["no user identity", "hard to rotate", "often leaked in logs/URLs"]},
    "mTLS": {"type": "mutual TLS", "transport": "TLS handshake", "pros": ["strong machine identity", "no tokens"], "cons": ["cert management complexity"]},
    "SAML": {"type": "XML-based SSO", "use": "enterprise SSO", "note": "Being replaced by OIDC in new systems"},
    "Basic Auth": {"type": "username:password in header", "encoding": "base64 (NOT encryption)", "security": "MUST use HTTPS", "use": "internal/simple APIs only"},
    "Bearer Token": {"type": "opaque or JWT", "transport": "Authorization: Bearer <token>", "standard": "RFC 6750"},
    "Passkeys/WebAuthn": {"type": "passwordless", "standard": "FIDO2/WebAuthn", "security": "phishing-resistant", "use": "modern passwordless auth"},
}

_VULNERABILITY_TYPES = {
    "XSS": {"full": "Cross-Site Scripting", "types": ["reflected", "stored", "DOM-based"], "impact": "session hijack, defacement, keylogging", "fix": "output encoding, CSP, sanitize input"},
    "CSRF": {"full": "Cross-Site Request Forgery", "impact": "unauthorized actions on behalf of user", "fix": "CSRF tokens, SameSite cookies, re-authentication for sensitive ops"},
    "SQL injection": {"impact": "data theft, modification, deletion, auth bypass", "fix": "parameterized queries, ORM, input validation, least-privilege DB user"},
    "IDOR": {"full": "Insecure Direct Object Reference", "impact": "access other users' data", "fix": "authorization checks on every request, use indirect references"},
    "SSRF": {"full": "Server-Side Request Forgery", "impact": "access internal services, cloud metadata", "fix": "URL allowlisting, block internal IPs, disable redirects"},
    "RCE": {"full": "Remote Code Execution", "impact": "full server compromise", "fix": "never eval user input, sandboxing, WAF, patch management"},
    "path traversal": {"impact": "read/write arbitrary files", "fix": "canonicalize paths, chroot, never concatenate user input into file paths"},
    "open redirect": {"impact": "phishing, token theft", "fix": "allowlist redirect URLs, validate against known origins"},
    "clickjacking": {"impact": "trick user into clicking hidden elements", "fix": "X-Frame-Options: DENY, CSP frame-ancestors 'none'"},
    "mass assignment": {"impact": "modify fields the user shouldn't (is_admin=true)", "fix": "explicit allowlists for assignable fields, DTOs"},
}


def owasp_top10(code: str) -> dict:
    """Get OWASP Top 10 entry by code (A01-A10)."""
    key = str(code).upper().strip()
    if not key.startswith("A"):
        key = f"A{key.zfill(2)}"
    entry = _OWASP_TOP_10.get(key)
    if not entry:
        return {"error": f"Unknown: {code}", "valid": sorted(_OWASP_TOP_10.keys())}
    return {"code": key, **entry}


def auth_method(name: str) -> dict:
    """Get details about an authentication method."""
    key = str(name).lower().strip()
    entry = _AUTH_METHODS.get(key)
    if not entry:
        for k, v in _AUTH_METHODS.items():
            if key in k.lower():
                return {"method": k, **v}
        return {"error": f"Unknown: {name}", "valid": sorted(_AUTH_METHODS.keys())}
    return {"method": key, **entry}


def vulnerability(name: str) -> dict:
    """Get details about a vulnerability type."""
    key = str(name).lower().strip()
    entry = _VULNERABILITY_TYPES.get(key)
    if not entry:
        for k, v in _VULNERABILITY_TYPES.items():
            if key in k.lower() or key == v.get("full", "").lower():
                return {"vulnerability": k, **v}
        return {"error": f"Unknown: {name}", "valid": sorted(_VULNERABILITY_TYPES.keys())}
    return {"vulnerability": key, **entry}


def jwt_vs_session() -> dict:
    """Compare JWT vs session-based authentication."""
    return {"JWT": _AUTH_METHODS["JWT"], "Session": _AUTH_METHODS["session"],
            "recommendation": "Sessions for traditional web apps, JWT for APIs/microservices/mobile"}


def list_owasp() -> list[dict]:
    """List all OWASP Top 10 2021 entries."""
    return [{"code": k, "name": v["name"]} for k, v in sorted(_OWASP_TOP_10.items())]


SECURITY_FUNCTIONS = {
    "owasp_top10": owasp_top10,
    "auth_method": auth_method,
    "vulnerability": vulnerability,
    "jwt_vs_session": jwt_vs_session,
    "list_owasp": list_owasp,
}

SECURITY_NL_PATTERNS = [
    (r'(?:what is|explain)\s+(?:OWASP\s+)?(A0[1-9]|A10)', 'owasp_top10("{0}")'),
    (r'(?:what is|explain)\s+(XSS|CSRF|SSRF|IDOR|RCE|SQL injection|clickjacking|open redirect|mass assignment|path traversal)', 'vulnerability("{0}")'),
    (r'(?:what is|explain|how does)\s+(JWT|OAuth2?|OIDC|SAML|mTLS|Basic Auth|session|API key|passkey|WebAuthn)\s+(?:auth|work)', 'auth_method("{0}")'),
    (r'(?:compare|difference|vs)\s+JWT\s+(?:and|vs)\s+session', 'jwt_vs_session()'),
    (r'(?:list|what are)\s+(?:the\s+)?OWASP\s+(?:top\s+)?10', 'list_owasp()'),
]
