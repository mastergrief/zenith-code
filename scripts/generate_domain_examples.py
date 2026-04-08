#!/usr/bin/env python3
"""Append hand-written domain examples to coding_reasoning_claude.jsonl.

Targets the topic gaps identified in the dataset analysis (2026-04-08): the
filtered HF data is heavily Python-biased and thin on real-world dev work like
DB/SQL, auth, DevOps, and web/API design. Each example follows the existing
format: system + user + assistant(<think>...</think> + answer), code blocks
required in the answer.

Run once:
    python3 scripts/generate_domain_examples.py
"""

from __future__ import annotations

import json
from pathlib import Path

TARGET = Path("agents/distill/data/coding_reasoning_claude.jsonl")
SYSTEM = "You are a helpful assistant"


def example(user: str, think: str, answer: str) -> dict:
    """Build one training example. Normalizes whitespace in think/answer."""
    assistant = f"<think>\n{think.strip()}\n</think>\n\n{answer.strip()}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant},
        ]
    }


EXAMPLES: list[dict] = []


# ── Auth ──────────────────────────────────────────────────────────

EXAMPLES.append(example(
    user="""I'm building a new web app and I can't decide between JWTs and server-side sessions. The whole "JWTs are stateless" thing sounds great but I keep reading they're a footgun. What should I actually do?""",
    think="""This is one of the most-asked questions and the answer has shifted in the last few years. The "JWTs are scalable because stateless" argument was overblown — most teams don't need it, and the downsides (you can't actually revoke a token without a server-side check, which defeats the stateless benefit) bite hard.

Real recommendation in 2026:
- Default to opaque session tokens stored server-side (Redis or database)
- The session ID is just a random string in a httpOnly cookie
- All the auth state lives server-side where you can revoke, audit, expire, log
- Use JWTs only for genuine cross-service or third-party scenarios where you can't make a server call

The "JWT for SPA" argument is mostly cargo cult — a session cookie works fine for an SPA, and an httpOnly cookie is more secure than localStorage anyway.

There are legitimate JWT use cases: short-lived service-to-service tokens, identity federation, mobile/desktop apps where you can't easily share cookies. But "I'm building a normal web app" isn't one.""",
    answer="""**Default to server-side sessions with an opaque token in a httpOnly cookie.** Use JWTs only if you have a specific reason that can't be solved another way.

**What "session" actually means here:**

```python
# On login
session_id = secrets.token_urlsafe(32)
redis.set(f"session:{session_id}", json.dumps({
    "user_id": user.id,
    "created_at": time.time(),
    "ip": request.ip,
}), ex=86400)  # 24h expiry

response.set_cookie(
    "session",
    session_id,
    httponly=True,    # JS cannot read it (blocks XSS exfiltration)
    secure=True,      # HTTPS only
    samesite="lax",   # CSRF mitigation
    max_age=86400,
)
```

On every request, look up `session:{cookie_value}` in Redis to get the user. If it's missing or expired, redirect to login.

**Why this is better than JWT for normal web apps:**

- **You can revoke tokens.** Delete the Redis key on logout, password change, or admin lockout. With JWTs you'd need a blocklist (defeating the "stateless" benefit) or wait for the token to expire.
- **You can update user data without re-issuing tokens.** Permissions, role, email — just update the user row. JWT claims are baked in until expiry.
- **You can audit and rate-limit per-session.** Track IP changes, concurrent sessions, anomalies.
- **The token is opaque** — no payload to leak if exposed. Can't be inspected client-side.
- **httpOnly cookie is XSS-safer than `localStorage`** — even if your JS gets compromised, the attacker can't read the token. (They can still piggyback requests, but exfiltration is harder.)

**When to actually use JWTs:**

- **Service-to-service auth** where one service can't call back to your auth server cheaply
- **Identity federation** (SAML/OIDC) — JWTs are the standard exchange format
- **Mobile/desktop apps** where cookies are awkward (use a refresh token in keychain, short-lived access JWT)
- **Public API tokens** that need to encode permissions without a DB hit per request

For all of these, prefer **short-lived** access tokens (15 min) with refresh tokens that DO hit the auth server.

**The "JWT is stateless and scales" myth:**

If your bottleneck is the millisecond cost of looking up a session by ID in Redis, you have a problem JWTs won't fix. A Redis SET on `session:abc123` is single-digit-microsecond fast and easily handles 100K+ RPS per node.

**Rule of thumb:** opaque session tokens in httpOnly cookies until you have a specific reason not to."""
))

EXAMPLES.append(example(
    user="""What bcrypt cost factor should I use? I see examples online with 10, 12, 14. How do I pick?""",
    think="""bcrypt cost factor is a log2 — cost 12 = 2^12 = 4096 rounds. Each +1 doubles the time.

The right cost is "as high as you can tolerate" balanced against your login throughput. Industry guidance:
- OWASP 2023+ recommends cost 10 minimum, prefers 12+
- Aim for ~250-500ms per hash on YOUR production hardware

Way to calibrate: time it on your server. Don't copy a number from a tutorial — your hardware is faster than the tutorial author's was 5 years ago.

Also worth flagging: bcrypt has a 72-byte input limit. Passwords longer than 72 bytes get silently truncated. Pre-hash with SHA-256 if you want to support arbitrary-length passwords (or use argon2 which doesn't have this limit).

argon2id is the modern preferred algorithm (won the 2015 Password Hashing Competition). bcrypt is still fine if you're already on it, but new code should default to argon2id.""",
    answer="""**Pick the highest cost where login takes ~250-500ms on YOUR production hardware.** Don't copy a number from a tutorial — hardware speeds change, and "cost 10" today isn't what it was in 2015.

**Calibrate it:**

```python
import bcrypt
import time

password = b"benchmark_password_123"
for cost in range(10, 16):
    start = time.perf_counter()
    bcrypt.hashpw(password, bcrypt.gensalt(rounds=cost))
    elapsed = (time.perf_counter() - start) * 1000
    print(f"cost={cost}: {elapsed:.0f}ms")
```

Run this on your production server (or an equivalent instance). Typical output on a modern cloud VM:
```
cost=10: 60ms
cost=11: 120ms
cost=12: 240ms
cost=13: 480ms    ← sweet spot for most apps
cost=14: 960ms
```

**Pick cost 12 or 13** for most apps. OWASP minimum is 10. Going higher than 14 starts hurting login UX without meaningful security gain (an attacker who's brute-forcing offline doesn't care if it's 480ms or 960ms — they'll throw GPUs at it).

**Don't forget the 72-byte limit:**

bcrypt silently truncates inputs longer than 72 bytes. If you support arbitrary-length passwords, pre-hash:

```python
import hashlib
import bcrypt
import base64

def hash_password(password: str) -> bytes:
    # SHA-256 → base64 to stay within bcrypt's 72-byte limit
    pre = base64.b64encode(hashlib.sha256(password.encode()).digest())
    return bcrypt.hashpw(pre, bcrypt.gensalt(rounds=12))

def verify_password(password: str, hashed: bytes) -> bool:
    pre = base64.b64encode(hashlib.sha256(password.encode()).digest())
    return bcrypt.checkpw(pre, hashed)
```

**For new code, prefer argon2id**:

```python
from argon2 import PasswordHasher

ph = PasswordHasher(time_cost=2, memory_cost=65536, parallelism=4)
hash = ph.hash("user_password")
ph.verify(hash, "user_password")  # raises on mismatch
```

argon2id won the Password Hashing Competition in 2015 and is the modern recommendation. It's memory-hard (resists GPU brute force better than bcrypt) and has no input length limit.

**Rule of thumb:** target ~300ms login time, use bcrypt cost 12 if you're already on bcrypt, use argon2id with default params for new projects."""
))

EXAMPLES.append(example(
    user="""How do I protect my SPA from CSRF? I'm using JWTs in localStorage and React on the frontend, FastAPI backend.""",
    think="""Two things to untangle here:

1. The user is using `localStorage` for JWT — that's an XSS vulnerability, not a CSRF one. Tokens in `localStorage` are NOT subject to CSRF (since they're sent via Authorization header, not cookies). But they ARE readable by any JS, so an XSS gives the attacker your token.

2. The actual CSRF protection question is moot for `Authorization: Bearer` tokens — CSRF only happens when the browser auto-attaches credentials (cookies, HTTP Basic). If your frontend explicitly attaches the token via JS, attackers from other origins can't replay it (they'd need to read it first, which is XSS).

So the right answer is: you don't have a CSRF problem with localStorage JWT, you have an XSS problem. The fix is to move to httpOnly cookies, which DOES introduce CSRF concerns, which you handle with SameSite cookies + a CSRF token.

This is one of those questions where the right answer is "you're solving the wrong problem".""",
    answer="""**You don't actually have a CSRF problem with `localStorage` JWTs.** CSRF requires the browser to auto-attach credentials (cookies, HTTP Basic), and `localStorage` tokens are explicitly attached by JS via `Authorization: Bearer ...`. An attacker on `evil.com` can't read your token from `localStorage` cross-origin, so they can't replay it.

**You have a different problem: XSS exfiltration.** `localStorage` is readable by ANY JavaScript on your page. If a third-party script gets injected (via a vulnerable dependency, an XSS, a compromised CDN), the attacker reads your token and impersonates the user.

**Real fix: move tokens out of `localStorage` and into httpOnly cookies. Then handle CSRF separately.**

**Step 1: Backend sets the token as a httpOnly cookie**

```python
# FastAPI
from fastapi import Response

@app.post("/login")
def login(creds: LoginCreds, response: Response):
    user = authenticate(creds)
    token = create_jwt(user.id)
    response.set_cookie(
        key="auth",
        value=token,
        httponly=True,
        secure=True,
        samesite="lax",  # blocks most CSRF automatically
        max_age=3600,
    )
    return {"user": user.public()}
```

**Step 2: Read the token from the cookie on each request**

```python
from fastapi import Cookie, HTTPException

def get_current_user(auth: str | None = Cookie(default=None)):
    if not auth:
        raise HTTPException(401, "not authenticated")
    return verify_jwt(auth)
```

**Step 3: Frontend stops attaching headers** — the cookie goes automatically:

```js
// Old way (delete this)
fetch("/api/me", {
  headers: { Authorization: `Bearer ${localStorage.getItem("token")}` }
});

// New way
fetch("/api/me", { credentials: "include" });
```

**Step 4: CSRF protection (now that you're using cookies)**

`SameSite=lax` blocks the most common CSRF attacks (it prevents the cookie from being sent on cross-site form posts). For sensitive state-changing endpoints, also use a CSRF token:

```python
# On login, also set a non-httpOnly CSRF token cookie
response.set_cookie(
    key="csrf",
    value=secrets.token_urlsafe(32),
    httponly=False,    # JS needs to read it to send it back
    secure=True,
    samesite="lax",
)
```

```js
// Frontend reads the cookie and sends it as a header
fetch("/api/transfer", {
  method: "POST",
  credentials: "include",
  headers: {
    "X-CSRF-Token": getCookie("csrf"),
    "Content-Type": "application/json",
  },
  body: JSON.stringify({ amount: 100 }),
});
```

```python
# Backend verifies the header matches the cookie
def csrf_protect(
    csrf_cookie: str = Cookie(alias="csrf"),
    csrf_header: str = Header(alias="X-CSRF-Token"),
):
    if csrf_cookie != csrf_header:
        raise HTTPException(403, "csrf mismatch")
```

This is the **double-submit cookie pattern**. It works because an attacker on `evil.com` can't read your `csrf` cookie (same-origin policy), so they can't construct a request with a matching header.

**Rule of thumb:** auth tokens belong in httpOnly cookies, not `localStorage`. SameSite=lax + double-submit CSRF tokens for state-changing endpoints."""
))

EXAMPLES.append(example(
    user="""Walk me through OAuth 2.0 with PKCE. I'm building a mobile app and the docs are confusing.""",
    think="""PKCE (Proof Key for Code Exchange) is the OAuth flow for "public" clients — apps that can't keep a secret (mobile, SPA, native desktop). The traditional OAuth code flow used a `client_secret` to authenticate the token exchange step, but a mobile app can't safely store one (it'd be in the binary, easily extracted).

PKCE replaces the client_secret with a one-time challenge generated per auth attempt:
1. App generates a random `code_verifier` (43-128 chars)
2. App computes `code_challenge = base64url(sha256(code_verifier))`
3. App sends `code_challenge` to the authorization endpoint
4. User logs in, IdP redirects back with an `authorization_code`
5. App sends the `authorization_code` AND the original `code_verifier` to the token endpoint
6. IdP verifies that `sha256(code_verifier)` matches the stored `code_challenge`, then issues tokens

This works because even if an attacker intercepts the authorization_code (via a malicious app registered for your scheme on Android, etc.), they don't have the `code_verifier` and can't exchange the code for tokens.

The user said "mobile app" so they need the redirect URL setup too (custom scheme or universal/app links).""",
    answer="""**PKCE (Proof Key for Code Exchange)** is the OAuth flow for clients that can't keep a secret — mobile apps, SPAs, desktop apps. It replaces the traditional `client_secret` with a per-login challenge so an intercepted auth code is useless to an attacker.

**The 6-step flow:**

```
1. App: generate random code_verifier
   code_verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"  (43-128 chars, URL-safe)

2. App: compute code_challenge = base64url(sha256(code_verifier))
   code_challenge = "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"

3. App: redirect user to IdP authorize endpoint with code_challenge
   GET https://auth.example.com/authorize?
       response_type=code&
       client_id=mobile_app&
       redirect_uri=myapp://callback&
       code_challenge=E9Melh...&
       code_challenge_method=S256&
       scope=read_profile&
       state=<random_csrf_token>

4. User authenticates → IdP redirects back to app:
   myapp://callback?code=AbCdEf123&state=<same_csrf_token>

5. App: POST to token endpoint with code AND original code_verifier
   POST https://auth.example.com/token
   Content-Type: application/x-www-form-urlencoded

   grant_type=authorization_code&
   code=AbCdEf123&
   redirect_uri=myapp://callback&
   client_id=mobile_app&
   code_verifier=dBjftJeZ4CVP...

6. IdP verifies sha256(code_verifier) == stored code_challenge,
   returns access_token + refresh_token
```

**Why it works:** even if an attacker intercepts the `code` (e.g., via a malicious app registered for `myapp://` on Android), they don't have the `code_verifier` and can't exchange the code for tokens.

**Implementation in Swift (iOS):**

```swift
import CryptoKit
import AuthenticationServices

// 1. Generate verifier and challenge
let verifier = generateCodeVerifier()  // 43-128 random URL-safe chars
let challenge = SHA256.hash(data: Data(verifier.utf8))
    .compactMap { String(format: "%02x", $0) }.joined()
    .base64URLEncoded()

// 2. Build authorize URL
var components = URLComponents(string: "https://auth.example.com/authorize")!
components.queryItems = [
    .init(name: "response_type", value: "code"),
    .init(name: "client_id", value: "mobile_app"),
    .init(name: "redirect_uri", value: "myapp://callback"),
    .init(name: "code_challenge", value: challenge),
    .init(name: "code_challenge_method", value: "S256"),
    .init(name: "scope", value: "read_profile"),
    .init(name: "state", value: UUID().uuidString),
]

// 3. Open ASWebAuthenticationSession (handles the browser redirect)
let session = ASWebAuthenticationSession(
    url: components.url!,
    callbackURLScheme: "myapp"
) { callbackURL, error in
    // 4. Extract code from callback
    let code = URLComponents(url: callbackURL!, resolvingAgainstBaseURL: false)!
        .queryItems!.first { $0.name == "code" }!.value!

    // 5. Exchange code + verifier for tokens
    exchangeCodeForToken(code: code, verifier: verifier)
}
session.start()
```

**Important details:**

- **Always check `state`** on the callback to prevent CSRF — it should match what you sent in step 3
- **`code_challenge_method=S256`** — never use `plain` (defeats the point)
- **Use the system browser** (`ASWebAuthenticationSession` on iOS, `Custom Tabs` on Android) — never an embedded webview, which lets your app see the user's password
- **Store refresh tokens in the secure enclave / keychain**, not `UserDefaults`
- **Universal Links / App Links are safer than custom schemes** — they're verified by the OS so other apps can't claim them

**Rule of thumb:** if you're building anything that can't safely store a `client_secret` (mobile, SPA, native), use PKCE. Don't use the implicit flow — it's deprecated and less secure than PKCE."""
))

EXAMPLES.append(example(
    user="""How do refresh tokens work and what's the right way to rotate them? I have 15-minute access tokens and 30-day refresh tokens.""",
    think="""Refresh tokens are long-lived credentials used to mint new access tokens without re-authenticating the user. The standard flow:

1. User logs in → server issues access_token (short-lived) + refresh_token (long-lived)
2. App uses access_token for API calls
3. Access token expires → app sends refresh_token to a `/refresh` endpoint
4. Server validates refresh_token → issues new access_token (and optionally new refresh_token)

The key security question is: what if a refresh token is stolen?

Two patterns:

**Static refresh tokens**: same token works until it expires (30 days). If stolen, attacker has 30 days of access. Used by many simple implementations.

**Refresh token rotation**: each /refresh call invalidates the old refresh token and issues a new one. If a stolen token is used, the legitimate user's NEXT refresh call will fail (token already used) — server detects this and revokes the entire token family. This is the OAuth 2.1 recommendation.

The "token family" concept is important: when you issue a new refresh token, link it to the old one. If a refresh token is reused (because the attacker used it AND the legit user tries), you can identify and revoke the whole chain.

Storage: refresh tokens in iOS keychain / Android keystore / httpOnly cookie. Never localStorage.""",
    answer="""**Use rotating refresh tokens with token family revocation.** Static refresh tokens (where the same token works for 30 days) leak access for 30 days if stolen — rotation lets you detect theft on the next legitimate refresh.

**Standard rotation flow:**

```
1. Login:
   server → app: {
     access_token: "...",     # 15 min, JWT
     refresh_token: "rt_001", # 30 days, opaque
     refresh_token_family: "fam_abc"
   }

2. Access token expires (15 min later):
   app → server: POST /refresh { refresh_token: "rt_001" }
   server: validate, issue new pair, mark rt_001 as USED
   server → app: {
     access_token: "...",
     refresh_token: "rt_002",  # NEW token
     refresh_token_family: "fam_abc"
   }

3. App stores rt_002, discards rt_001
4. Next refresh uses rt_002 → issues rt_003, marks rt_002 as USED
```

**The theft detection:**

```
1. Attacker steals rt_002 (e.g., via XSS, network capture, malware)
2. Attacker uses rt_002 → server issues rt_003 to attacker
3. Legit user later tries to refresh with rt_002 (their stored token)
4. Server sees rt_002 is already USED → ALERT: refresh token reuse!
5. Server revokes the entire family fam_abc → both legit user and attacker are logged out
6. Legit user is forced to re-authenticate (which is correct — their device may be compromised)
```

**Server-side schema:**

```sql
CREATE TABLE refresh_tokens (
  token_id TEXT PRIMARY KEY,        -- the opaque token value
  family_id TEXT NOT NULL,           -- groups rotated tokens
  user_id BIGINT NOT NULL,
  issued_at TIMESTAMPTZ NOT NULL,
  expires_at TIMESTAMPTZ NOT NULL,
  used_at TIMESTAMPTZ,               -- NULL until consumed
  revoked BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_rt_family ON refresh_tokens (family_id);
CREATE INDEX idx_rt_user ON refresh_tokens (user_id);
```

**Refresh endpoint:**

```python
@app.post("/refresh")
def refresh(refresh_token: str = Cookie(...)):
    rt = db.fetch_one(
        "SELECT * FROM refresh_tokens WHERE token_id = %s",
        (refresh_token,)
    )
    if not rt:
        raise HTTPException(401, "invalid token")
    if rt.revoked or rt.expires_at < now():
        raise HTTPException(401, "expired or revoked")

    if rt.used_at is not None:
        # REUSE DETECTED — revoke entire family
        db.execute(
            "UPDATE refresh_tokens SET revoked = TRUE WHERE family_id = %s",
            (rt.family_id,)
        )
        log.warn(f"refresh token reuse detected for user {rt.user_id}")
        raise HTTPException(401, "token reuse — please re-authenticate")

    # Mark old token as used
    db.execute(
        "UPDATE refresh_tokens SET used_at = NOW() WHERE token_id = %s",
        (refresh_token,)
    )

    # Issue new pair in the same family
    new_rt = secrets.token_urlsafe(32)
    db.execute(
        "INSERT INTO refresh_tokens (token_id, family_id, user_id, issued_at, expires_at) "
        "VALUES (%s, %s, %s, NOW(), NOW() + INTERVAL '30 days')",
        (new_rt, rt.family_id, rt.user_id)
    )

    return {
        "access_token": create_jwt(rt.user_id, expires_in=900),
        "refresh_token": new_rt,
    }
```

**Client storage:**
- iOS: Keychain (`kSecClassGenericPassword`, `kSecAttrAccessibleAfterFirstUnlock`)
- Android: EncryptedSharedPreferences or Android Keystore
- Web: httpOnly cookie with `SameSite=strict`
- **Never** `localStorage`, `sessionStorage`, or `AsyncStorage`

**Rule of thumb:** rotate refresh tokens on every use, group them in families, revoke the whole family on detected reuse. 15-minute access tokens are reasonable; 30-day refresh is on the long side — consider 7-14 days unless you have a clear reason."""
))

EXAMPLES.append(example(
    user="""I want to rate-limit my login endpoint to prevent credential stuffing. How aggressive should I be without locking out real users?""",
    think="""Login rate limiting is about catching credential stuffing (attacker tries leaked username/password pairs from a breach) and brute force (trying many passwords for one user). But you can't be so aggressive you lock real users out — they fat-finger their password sometimes.

Two-dimensional rate limit:
1. **Per IP** — catches widely-distributed credential stuffing where attacker tries many user/pass combos from one source
2. **Per username** — catches focused brute force on one account

Standard limits:
- Per IP: 10 failed attempts per minute, then block for 15 min
- Per username: 5 failed attempts per 15 min, then require CAPTCHA or email confirmation
- After 20 failed attempts on a username: lock account for 1 hour, email user

The CAPTCHA-after-N approach is better than hard blocks because it catches automated attacks but lets real users continue.

Also: only count FAILED attempts, not successful ones. And reset the counter on successful login. And add jitter to the lockout to prevent timing attacks.

Important: rate limit on the login endpoint specifically, not all endpoints. And don't rate-limit signed-in users (different problem).""",
    answer="""**Two-dimensional rate limit: per IP and per username.** Per-IP catches credential stuffing (one attacker, many accounts). Per-username catches focused brute force (one account, many passwords). Both are needed.

**Suggested thresholds (tune for your traffic):**

| Layer | Threshold | Action |
|---|---|---|
| Per IP | 10 failed attempts / min | Block IP for 15 min |
| Per username | 5 failed attempts / 15 min | Require CAPTCHA |
| Per username | 20 failed attempts / hour | Lock account 1 hour, email user |

**Implementation with Redis:**

```python
import redis
import time

r = redis.Redis()

def check_login_rate_limit(ip: str, username: str) -> tuple[bool, str]:
    # Returns (allowed, reason). Call BEFORE checking the password.
    # Per-IP check
    ip_key = f"login:ip:{ip}"
    ip_count = r.get(ip_key)
    if ip_count and int(ip_count) >= 10:
        return False, "too_many_attempts_from_ip"

    # Per-username check
    user_key = f"login:user:{username.lower()}"
    user_count = r.get(user_key)
    if user_count and int(user_count) >= 5:
        return False, "captcha_required"
    if user_count and int(user_count) >= 20:
        return False, "account_locked"

    return True, "ok"


def record_failed_login(ip: str, username: str):
    # Call AFTER a password check fails.
    ip_key = f"login:ip:{ip}"
    r.incr(ip_key)
    r.expire(ip_key, 60)  # 1 min window

    user_key = f"login:user:{username.lower()}"
    r.incr(user_key)
    r.expire(user_key, 900)  # 15 min window


def record_successful_login(ip: str, username: str):
    # Reset counters on success.
    r.delete(f"login:ip:{ip}", f"login:user:{username.lower()}")
```

**In the login handler:**

```python
@app.post("/login")
def login(creds: LoginCreds, request: Request):
    ip = request.client.host

    allowed, reason = check_login_rate_limit(ip, creds.username)
    if not allowed:
        if reason == "captcha_required":
            return {"error": "captcha", "site_key": CAPTCHA_SITE_KEY}, 429
        if reason == "account_locked":
            send_account_locked_email(creds.username)  # async
            return {"error": "account_locked"}, 429
        return {"error": "rate_limited"}, 429

    user = db.find_user(creds.username)
    # IMPORTANT: same response time whether user exists or not (timing attack)
    if not user or not verify_password(creds.password, user.password_hash):
        record_failed_login(ip, creds.username)
        return {"error": "invalid_credentials"}, 401

    record_successful_login(ip, creds.username)
    return create_session(user)
```

**Key details:**

- **Only count FAILED attempts.** Don't penalize a user who logs in successfully then logs in again later.
- **Reset on success.** Otherwise a user with a bad day's typing gets locked out tomorrow.
- **Same error response and timing** for "user doesn't exist" vs "wrong password" — leaking which is which lets attackers enumerate usernames.
- **Lowercase the username** for the rate limit key (`Alice@example.com` and `alice@example.com` should hit the same bucket).
- **Add a CAPTCHA tier** before the hard lockout — bots fail CAPTCHA, real users solve it and continue.
- **Don't lock the account based on IP attempts** — that's a denial-of-service vector (attacker locks every user by hammering their account from random IPs).

**Rule of thumb:** rate limit progressively (warn → CAPTCHA → temporary lock → permanent lock with email recovery), and always have a path back for legitimate users."""
))

EXAMPLES.append(example(
    user="""I want to add magic-link login (passwordless email-only). Walk me through the security considerations.""",
    think="""Magic links have specific failure modes:
1. Replay attacks — same link works multiple times → use single-use tokens
2. Long-lived links — old emails work forever → expire after 5-15 minutes
3. Session fixation — attacker generates a link for victim's email, sends it to themselves → use a "verify same browser" cookie
4. Email forwarding — user accidentally forwards the email → token revoked on first use
5. Open redirect — `?next=https://evil.com` → whitelist the next param
6. Token in URL = leaks via referer/logs → use POST or fragment

Storage:
- Generate a high-entropy token (32+ random bytes)
- Hash it before storing (so DB compromise doesn't grant access)
- Store: hash, user_id, expires_at, used_at, browser_fingerprint

Flow:
1. User enters email
2. Server: generate token, store hash, send email with link
3. User clicks link → server validates, marks used, creates session
4. Same response for "email exists" and "email doesn't exist" — don't leak

Add rate limiting too — 1 link request per email per minute, max 5 per hour.""",
    answer="""**Magic links have specific failure modes — single-use, short expiry, and same-browser verification handle the worst ones.** Here's the full flow with security details inline.

**Database schema:**

```sql
CREATE TABLE magic_links (
  token_hash TEXT PRIMARY KEY,        -- sha256 of the actual token
  user_email TEXT NOT NULL,
  browser_fingerprint TEXT NOT NULL,  -- cookie set when link was requested
  created_at TIMESTAMPTZ NOT NULL,
  expires_at TIMESTAMPTZ NOT NULL,
  used_at TIMESTAMPTZ                 -- NULL until consumed
);

CREATE INDEX idx_ml_email ON magic_links (user_email);
```

**Step 1: Request a link**

```python
import secrets
import hashlib

@app.post("/auth/request-link")
def request_link(email: str, response: Response):
    # Rate limit: 1 per email per minute
    if not rate_limiter.allow(f"link:{email.lower()}", limit=1, window=60):
        return {"status": "ok"}, 200  # ← same response either way (don't leak)

    # Generate high-entropy token
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode()).hexdigest()

    # Set a browser fingerprint cookie BEFORE sending the email
    fingerprint = secrets.token_urlsafe(32)
    response.set_cookie(
        "ml_fp",
        fingerprint,
        httponly=True,
        secure=True,
        samesite="strict",
        max_age=900,  # 15 min, same as link expiry
    )

    # Store the hash (not the token itself)
    db.execute(
        "INSERT INTO magic_links (token_hash, user_email, browser_fingerprint, created_at, expires_at) "
        "VALUES (%s, %s, %s, NOW(), NOW() + INTERVAL '15 minutes')",
        (token_hash, email.lower(), fingerprint)
    )

    # Send the email — link goes to GET /auth/verify?token=...
    send_email(
        to=email,
        subject="Sign in to MyApp",
        body=f"Click to sign in: https://myapp.com/auth/verify?token={token}\n\nThis link expires in 15 minutes."
    )

    # Same response whether email exists or not — don't leak
    return {"status": "ok"}, 200
```

**Step 2: Verify the link**

```python
@app.get("/auth/verify")
def verify(token: str, ml_fp: str = Cookie(default=None)):
    token_hash = hashlib.sha256(token.encode()).hexdigest()

    link = db.fetch_one(
        "SELECT * FROM magic_links WHERE token_hash = %s",
        (token_hash,)
    )
    if not link:
        return {"error": "invalid"}, 400
    if link.used_at:
        return {"error": "already_used"}, 400
    if link.expires_at < now():
        return {"error": "expired"}, 400

    # Same-browser check: prevents email-forward attacks
    if link.browser_fingerprint != ml_fp:
        return {"error": "different_browser", "message": "Open this link in the browser where you requested it."}, 400

    # Mark used (atomic — guards against double-click race)
    updated = db.execute(
        "UPDATE magic_links SET used_at = NOW() WHERE token_hash = %s AND used_at IS NULL",
        (token_hash,)
    )
    if updated.rowcount == 0:
        return {"error": "race_condition"}, 400

    user = db.find_or_create_user(link.user_email)
    return create_session(user)
```

**The security details that matter:**

- **Hash the token before storing** — DB compromise doesn't grant immediate access to active links
- **Single-use** — `used_at` is set on first verify; double-click or replay fails
- **Short expiry** — 15 min is a good balance (long enough for slow inboxes, short enough that stale emails are useless)
- **Same-browser fingerprint** — attacker can't just request a link for victim's email and use it themselves; they'd need the cookie too
- **Same response on success/failure** — `/auth/request-link` returns 200 whether the email exists or not, preventing user enumeration
- **Rate limit** — both per-email (1/min, 5/hour) and per-IP (10/hour)
- **`SameSite=strict` cookie** — prevents the fingerprint from leaking on cross-site navigations
- **Use POST for the verify** if you want to avoid token-in-URL referer leaks (but breaks email client UX). Most apps accept the tradeoff.
- **Don't auto-create accounts on first link** if you have an account verification flow — link login implies email ownership, but you may want to ask for additional info first

**Rule of thumb:** magic links are great UX but require defense in depth. Single-use + short expiry + same-browser are the minimums; rate limiting and timing-equal responses are essential additions."""
))

EXAMPLES.append(example(
    user="""Where should I store my API keys and database credentials? I keep seeing different advice — env vars, .env files, vaults...""",
    think="""This is a common confusion. The hierarchy of "where to store secrets" matters:

1. **Worst**: hardcoded in source. Accidentally committed, anyone with repo access has them, can't rotate without redeploy.
2. **Bad**: `.env` file in repo. Same problems plus you've made it look intentional.
3. **OK for dev**: `.env.local` in `.gitignore`. Safe locally if you're disciplined, but easy to accidentally commit.
4. **Good for most**: environment variables injected at runtime by the orchestrator. The app reads `os.environ`, the orchestrator (Docker, k8s, ECS, systemd) sets them from a secret store.
5. **Best**: secrets manager (AWS Secrets Manager, GCP Secret Manager, HashiCorp Vault, Doppler) — fetched at startup, supports rotation, audit logs, fine-grained IAM, encryption at rest.

The key insight: the "where" (env var vs vault) is less important than "rotated, scoped, auditable, never in source control". Env vars in production are fine if your orchestrator injects them from a vault.

Local dev: `.env.local` (gitignored) is fine for personal dev databases. Don't put production credentials there.

Common mistake: setting prod env vars in CI without a secret store. They show up in build logs.""",
    answer="""**Tiered approach**: secrets manager for production, environment variables (injected by your orchestrator) at runtime, `.env.local` (gitignored) only for local dev.

**The ranking (worst to best):**

| Approach | Verdict |
|---|---|
| Hardcoded in source | NEVER. Accidentally committed = permanent leak. |
| `.env` checked in | NEVER. Same as hardcoded. |
| `.env.local` gitignored | OK for **local dev only** (personal DB credentials). Easy to fat-finger and commit. |
| `process.env.X` set by orchestrator | Good. Industry standard for production. |
| Secrets manager (Vault, AWS SM, GCP SM) | Best. Rotation, audit, IAM scoping. |

**Local dev — use a gitignored `.env.local`:**

```
# .gitignore
.env.local
.env.*.local
```

```bash
# .env.local
DATABASE_URL=postgres://localhost:5432/myapp_dev
STRIPE_KEY=sk_test_...   # test key, not live
SESSION_SECRET=dev_only_not_production
```

```python
# Your app reads it via dotenv (dev) or just os.environ (prod)
from dotenv import load_dotenv
import os

if os.environ.get("ENV") != "production":
    load_dotenv(".env.local")

DATABASE_URL = os.environ["DATABASE_URL"]
```

**Production — secrets manager + orchestrator injection:**

The pattern: secrets live in the secrets manager. Your orchestrator fetches them at deploy/start time and injects as environment variables. The app code stays the same (`os.environ`).

**AWS example (ECS task definition):**

```json
{
  "containerDefinitions": [{
    "name": "api",
    "image": "myapp:v123",
    "secrets": [
      {
        "name": "DATABASE_URL",
        "valueFrom": "arn:aws:secretsmanager:us-east-1:123:secret:prod/db_url-AbCdEf"
      },
      {
        "name": "STRIPE_KEY",
        "valueFrom": "arn:aws:secretsmanager:us-east-1:123:secret:prod/stripe-XyZw"
      }
    ]
  }]
}
```

ECS fetches the secret at task start, sets it as an env var, and your app reads `os.environ["DATABASE_URL"]` like normal. The secret never appears in your task definition, your image, or your CI logs.

**Kubernetes — use Sealed Secrets, External Secrets Operator, or Vault Agent:**

```yaml
# external-secrets.io
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: app-secrets
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: SecretStore
  target:
    name: app-secrets
  data:
    - secretKey: DATABASE_URL
      remoteRef:
        key: prod/db_url
```

The operator pulls from AWS Secrets Manager into a k8s Secret, your pods mount that as env vars.

**Things to also do:**

- **Different secrets per environment** — never share prod secrets with dev/staging. Use separate vault paths.
- **Rotate regularly** — secrets managers support automatic rotation for many integrations (RDS, Redis, etc.)
- **Audit access** — who fetched which secret when. AWS CloudTrail / GCP audit logs cover this.
- **Scope IAM tightly** — the API service can read `prod/api/*` but not `prod/payments/*`. Least privilege.
- **Pre-commit hooks** to catch accidental commits:
  ```bash
  # .pre-commit-config.yaml
  - repo: https://github.com/Yelp/detect-secrets
    hooks:
      - id: detect-secrets
  ```
- **Truffle Hog or git-secrets** to scan history if you suspect a leak

**If a secret leaks:**

1. **Rotate immediately** — don't wait to investigate
2. Audit logs to understand blast radius
3. **Revoke + re-issue** rather than removing the leaked value (the leaked one may still work in caches)
4. Force re-deploy with new secret
5. Post-mortem the leak path (how did it get committed? was a check missing?)

**Rule of thumb:** if your secret can be retrieved from a place a developer can `cat`, you're not using a secrets manager — you're using a fancier `.env` file."""
))

# ── DB / SQL ──────────────────────────────────────────────────────

EXAMPLES.append(example(
    user="""My Rails app is suddenly hitting the DB hundreds of times per page load. I'm rendering a list of orders and calling order.user.name in the view. What's going on?""",
    think="""Classic N+1. The view iterates orders, and each `order.user` access lazy-loads that user row — 100 orders → 1 query for orders + 100 queries for users.

Fix: eager-load with `includes(:user)`. That makes Rails fire two queries total (orders + a single `WHERE id IN (...)` for users).

Nuance:
- `includes` does separate queries (default, fast for big result sets)
- `joins` does INNER JOIN — only use if you need to filter on the joined table
- If you `where` references the associated table, add `.references(:user)` so Rails knows to JOIN
- `bullet` gem catches N+1 in dev""",
    answer="""**Classic N+1 query problem.** Each `order.user` triggers a separate SELECT for that user — 100 orders means 101 queries (1 + 100).

**Fix: eager-load the association**

```ruby
@orders = Order.includes(:user).where(status: 'pending')
```

Now Rails fires two queries:
```sql
SELECT * FROM orders WHERE status = 'pending';
SELECT * FROM users WHERE id IN (1, 2, 3, ...);
```

**`includes` vs `joins`:**
- `includes` → separate queries (faster for large result sets, default)
- `joins` → INNER JOIN (use when you need to filter or aggregate on the joined table)
- If your `where` clause references `users.something`, use `includes(:user).references(:user)` so Rails generates the JOIN

**Catch it in dev:**
```ruby
# Gemfile
gem 'bullet', group: :development
```

Bullet logs warnings on every N+1 it detects.

**Rule of thumb:** if you're iterating a collection in a view or serializer and accessing an association, you almost certainly need `includes`."""
))

EXAMPLES.append(example(
    user="""EXPLAIN ANALYZE on this query shows a Seq Scan and it's taking 8 seconds:
```sql
SELECT * FROM events WHERE user_id = 12345 AND created_at > NOW() - INTERVAL '7 days';
```
The events table has 50M rows. There's an index on user_id. Why isn't it using it?""",
    think="""The user has an index on `user_id` alone but the query also filters on `created_at`. Postgres has options:

1. Use the `user_id` index, then filter by `created_at` after fetching → fast if user has few events, slow if they have many
2. Seq scan the whole table → almost never wins on 50M rows unless the planner thinks the user_id index is too selective in the wrong direction

The fact that it's choosing seq scan suggests the planner thinks the user_id index would return too many rows to be worth using (high cardinality, or stale stats). Or the index is on the wrong column type.

The clean fix is a composite index on `(user_id, created_at DESC)`. This serves the exact query: index lookup by user_id, then range scan within that user's events ordered by date. Postgres can stop scanning as soon as it crosses the 7-day boundary.

Other things to check:
- `ANALYZE events` — refresh statistics. Stale stats cause wrong plans.
- Is `user_id` actually the same type in the table and the query? An int filter against a bigint column may not use the index.""",
    answer="""**The single-column `user_id` index isn't selective enough** for the planner — it's choosing to seq-scan instead. With 50M rows, you need a composite index that covers both predicates.

**Fix: create a composite index**

```sql
CREATE INDEX CONCURRENTLY idx_events_user_created
  ON events (user_id, created_at DESC);
```

`CONCURRENTLY` is important on a 50M row table — without it, you'll lock writes for the duration of the build.

This index lets Postgres:
1. Jump to the user's rows via the `user_id` portion
2. Range-scan within those rows ordered by `created_at`
3. Stop as soon as it crosses the 7-day boundary

The DESC ordering matches your typical "recent events first" use case.

**Re-run EXPLAIN ANALYZE after the index builds:**
```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM events WHERE user_id = 12345 AND created_at > NOW() - INTERVAL '7 days';
```

You should see `Index Scan using idx_events_user_created` and a much smaller `actual time`.

**Also worth checking:**

```sql
ANALYZE events;  -- refresh table statistics so the planner has accurate row counts
```

Stale stats are a common reason the planner picks the wrong plan, especially on tables that just had a big insert.

**Rule of thumb:** when a query has `WHERE a = ? AND b > ?`, the right index is `(a, b)` — equality predicates first, range predicates second. Single-column indexes only help when one column is highly selective on its own."""
))

EXAMPLES.append(example(
    user="""Is this Python query safe?
```python
def get_user(username):
    cursor.execute(f"SELECT * FROM users WHERE name = '{username}'")
    return cursor.fetchone()
```
The username comes from a form field.""",
    think="""This is textbook SQL injection. `f"...{username}..."` interpolates user input directly into the query string. An attacker submits `' OR '1'='1` as the username and the query becomes `SELECT * FROM users WHERE name = '' OR '1'='1'` — returns every user. Or worse, they submit `'; DROP TABLE users; --` and you lose the table.

The fix is parameterized queries — pass the value as a separate argument so the driver handles escaping. Every Python DB driver supports this. The placeholder syntax depends on the driver (`?` for sqlite, `%s` for psycopg/mysqlclient, `:name` for sqlalchemy).

I should also flag the `*` selection (return only the columns you need) but the SQL injection is the urgent thing.""",
    answer="""**No — that's textbook SQL injection.** An attacker submits `' OR '1'='1` as the username and your query becomes:

```sql
SELECT * FROM users WHERE name = '' OR '1'='1'
```

That returns every user. Worse, `'; DROP TABLE users; --` will silently drop your users table.

**Fix: parameterized query.** Pass the value as a separate argument so the driver handles escaping:

```python
def get_user(username):
    cursor.execute(
        "SELECT id, name, email FROM users WHERE name = %s",
        (username,)
    )
    return cursor.fetchone()
```

The placeholder syntax depends on your driver:
- `psycopg2` / `psycopg3` (Postgres): `%s`
- `sqlite3`: `?`
- `mysql.connector`: `%s`
- SQLAlchemy: `:name` with a dict, or use the ORM

**Important: `%s` is NOT Python string formatting here** — don't write `cursor.execute("... %s" % username)`. That's still injection. The placeholder gets passed as a separate argument to `execute()`.

**With SQLAlchemy ORM (preferred for new code):**

```python
def get_user(username):
    return session.query(User).filter(User.name == username).first()
```

The ORM handles parameterization automatically.

**While you're here:** also stop using `SELECT *`. List the columns you actually need (`SELECT id, name, email`) — it's faster, safer if the schema changes, and makes the intent clear.

**Rule of thumb:** if you're concatenating, formatting, or interpolating user input into a SQL string, you have a SQL injection bug. Always pass values as parameters."""
))

EXAMPLES.append(example(
    user="""I need to insert-or-update a row in Postgres. Right now I'm doing SELECT, then INSERT or UPDATE based on whether the row exists. Two clients can hit the SELECT at the same time and both think the row doesn't exist. What's the right pattern?""",
    think="""Classic check-then-act race condition. Two clients SELECT, both see "not found", both INSERT — and one fails on the unique constraint (or worse, you get a duplicate if there's no constraint).

Postgres has `INSERT ... ON CONFLICT` (UPSERT) for exactly this. It's atomic — one statement, no race.

Two flavors:
- `ON CONFLICT DO NOTHING` — insert if absent, ignore if present
- `ON CONFLICT (col) DO UPDATE SET ...` — insert if absent, update if present

For the update form, you reference the would-be-inserted row as `EXCLUDED.col`. So `EXCLUDED.email` is the email you tried to insert.

Important: there must be a UNIQUE constraint or unique index on the column(s) you're conflicting on. Otherwise Postgres has nothing to detect the conflict against.""",
    answer="""**Don't roll your own check-then-act** — use Postgres `INSERT ... ON CONFLICT` (UPSERT). It's atomic, so the race goes away.

**For "insert or update":**

```sql
INSERT INTO users (email, name, last_seen)
VALUES ('alice@example.com', 'Alice', NOW())
ON CONFLICT (email) DO UPDATE
  SET name = EXCLUDED.name,
      last_seen = EXCLUDED.last_seen;
```

`EXCLUDED.column` refers to the row you tried to insert. The whole statement is one atomic operation — no race window.

**For "insert only if not present" (no update):**

```sql
INSERT INTO users (email, name)
VALUES ('alice@example.com', 'Alice')
ON CONFLICT (email) DO NOTHING;
```

**Prerequisite:** there must be a UNIQUE constraint or unique index on the conflict target column. Without it, Postgres has nothing to detect the conflict against:

```sql
ALTER TABLE users ADD CONSTRAINT users_email_unique UNIQUE (email);
```

**Conditional update** — only update if a condition holds:

```sql
INSERT INTO sessions (user_id, last_active)
VALUES (123, NOW())
ON CONFLICT (user_id) DO UPDATE
  SET last_active = EXCLUDED.last_active
  WHERE sessions.last_active < EXCLUDED.last_active;
```

This avoids overwriting a newer value with an older one (useful for distributed clients with clock skew).

**Rule of thumb:** any time your code does `SELECT, then INSERT or UPDATE based on the result`, you have a TOCTOU (time-of-check-to-time-of-use) race. Use UPSERT, or wrap in a transaction with `SELECT ... FOR UPDATE`."""
))

EXAMPLES.append(example(
    user="""I need to add a NOT NULL column with a default to a 50M row Postgres table in production. Last time I tried this it locked the table for 20 minutes. What's the safe approach?""",
    think="""Adding a NOT NULL column with a default used to rewrite the whole table — Postgres had to fill in the default for every existing row, which acquires an ACCESS EXCLUSIVE lock the entire time. On 50M rows, that's minutes of downtime.

Postgres 11+ changed this for SIMPLE defaults — it stores the default in catalog and only writes new rows. So `ALTER TABLE foo ADD COLUMN bar TEXT NOT NULL DEFAULT 'x'` is fast on PG11+ as long as the default is a constant, not a volatile expression.

But for safety, the bulletproof pattern works on all versions:
1. Add the column nullable, no default
2. Backfill in batches (avoid one giant UPDATE that locks)
3. Set the default
4. Add NOT NULL constraint via NOT VALID + VALIDATE (which only takes a brief lock)

The user said they got bitten by this before, so they're probably on an older Postgres or hit the volatile default case. Best to give the bulletproof pattern.""",
    answer="""**The bulletproof multi-step pattern works on all Postgres versions.** It avoids the long table-rewrite by splitting the change into small operations.

**Step 1: Add column nullable, no default** (instant — just a metadata change):
```sql
ALTER TABLE events ADD COLUMN tenant_id BIGINT;
```

**Step 2: Backfill in batches** (avoids one giant UPDATE locking the table):
```sql
DO $$
DECLARE
  batch_size INT := 10000;
  rows_updated INT;
BEGIN
  LOOP
    UPDATE events
    SET tenant_id = 0
    WHERE id IN (
      SELECT id FROM events WHERE tenant_id IS NULL LIMIT batch_size
    );
    GET DIAGNOSTICS rows_updated = ROW_COUNT;
    EXIT WHEN rows_updated = 0;
    PERFORM pg_sleep(0.1);  -- small breather between batches
  END LOOP;
END $$;
```

(In application code, you'd do this in a script with explicit pagination — same idea.)

**Step 3: Set the default for new rows** (instant, just stores the default):
```sql
ALTER TABLE events ALTER COLUMN tenant_id SET DEFAULT 0;
```

**Step 4: Add NOT NULL constraint via NOT VALID + VALIDATE** (avoids long lock):
```sql
ALTER TABLE events ADD CONSTRAINT events_tenant_id_not_null
  CHECK (tenant_id IS NOT NULL) NOT VALID;

ALTER TABLE events VALIDATE CONSTRAINT events_tenant_id_not_null;
```

`NOT VALID` skips the existing-row check (instant). `VALIDATE` then scans the table without taking an exclusive lock — readers and writers continue normally.

**Step 5 (optional): Convert CHECK constraint to NOT NULL** (briefly locks table):
```sql
ALTER TABLE events ALTER COLUMN tenant_id SET NOT NULL;
ALTER TABLE events DROP CONSTRAINT events_tenant_id_not_null;
```

Postgres 12+ recognizes the CHECK constraint and skips re-validation, so this is fast. On older versions, leave the CHECK constraint as-is — it has the same effect.

**Postgres 11+ shortcut:** If you have PG11+ AND your default is a simple constant (not `NOW()` or anything volatile), this whole thing collapses to:
```sql
ALTER TABLE events ADD COLUMN tenant_id BIGINT NOT NULL DEFAULT 0;
```
PG11 stores the default in catalog and only writes it for new rows — instant on any table size.

**Rule of thumb:** for any schema change on a large production table, assume the worst — write the multi-step version. The "fast path" is a happy bonus when it works."""
))


EXAMPLES.append(example(
    user="""My API endpoint paginates with `LIMIT 50 OFFSET ?`. It's fast on page 1 but takes 8 seconds on page 1000. Why and how do I fix it?""",
    think="""OFFSET is O(N) — Postgres still has to scan and discard the first OFFSET rows before returning the LIMIT. At OFFSET 50000 it scans 50050 rows just to return 50.

The fix is cursor pagination (also called keyset pagination). Instead of "page N", you pass the last item from the previous page and use it as a WHERE filter. The query becomes a fast index seek.

Tradeoffs:
- OFFSET: lets you jump to arbitrary pages (page 47), shows total count easily, slow on deep pages
- Cursor: only forward/backward navigation, no easy total count, fast at any depth

For infinite scroll / "load more" UIs, cursor is strictly better. For numbered page UIs ("Page 1, 2, 3..."), you may need OFFSET or a hybrid.

The cursor needs to be on a stable, indexed, unique column (or column tuple). `id` works if it's monotonic. For sort-by-date, use `(created_at, id)` to break ties.""",
    answer="""**OFFSET is O(N).** At page 1000 with `LIMIT 50 OFFSET 50000`, Postgres scans and discards 50,000 rows before returning 50. Page depth means linear time.

**Fix: cursor (keyset) pagination.** Pass the last item from the previous page as a `WHERE` filter — the query becomes a fast index seek.

**Sorted by `id`:**
```sql
SELECT id, title, created_at FROM posts
WHERE id > $1            -- $1 is the last id from the previous page
ORDER BY id
LIMIT 50;
```

**Sorted by `created_at` (with tiebreaker for non-unique timestamps):**
```sql
SELECT id, title, created_at FROM posts
WHERE (created_at, id) > ($1, $2)
ORDER BY created_at, id
LIMIT 50;
```

Make sure you have an index on the sort columns:
```sql
CREATE INDEX idx_posts_created_id ON posts (created_at, id);
```

**Client side**, the cursor is the `id` (or `created_at + id`) of the last row in the current page:
```python
def get_page(cursor=None, limit=50):
    if cursor is None:
        rows = db.execute("SELECT ... ORDER BY id LIMIT %s", (limit,))
    else:
        rows = db.execute(
            "SELECT ... WHERE id > %s ORDER BY id LIMIT %s",
            (cursor, limit)
        )
    next_cursor = rows[-1].id if rows else None
    return {"items": rows, "next_cursor": next_cursor}
```

**Tradeoffs:**

| | OFFSET | Cursor |
|---|---|---|
| Deep pages | Slow (O(N)) | Fast (O(log N)) |
| Jump to arbitrary page | Easy | Hard |
| Total count | Easy | Hard (separate query) |
| New items shifting pages | Yes (annoying) | No |

**Rule of thumb:** if your UI is infinite scroll or "load more", use cursor. If it's numbered pages with "go to page 47", use OFFSET but cap the max page (e.g., redirect to search if user wants page > 100)."""
))

EXAMPLES.append(example(
    user="""I'm storing JSON config in a JSONB column and querying with `WHERE config->>'env' = 'prod'`. It does a full table scan even though I have a BTREE index on `config`. What index should I use?""",
    think="""BTREE on a JSONB column indexes the entire JSONB value as a unit — useful for equality on the whole document, useless for querying nested keys.

For querying inside JSONB, you have two index options:

1. **GIN index on the whole column**: `CREATE INDEX ... USING gin (config)` — supports `?`, `?|`, `?&`, `@>`, `@?`, `@@` operators. Larger index, slower writes, but flexible: works for any key.

2. **BTREE index on a specific expression**: `CREATE INDEX ... ON foo ((config->>'env'))` — small, fast, but only helps queries on that exact key. You'd need one per key you want to query.

The user's query (`config->>'env' = 'prod'`) is a single-key equality. The expression index is the right call — it's smaller and the query plan will use it directly. They could also rewrite the query to use the containment operator and a GIN index.""",
    answer="""**BTREE on the whole JSONB column doesn't help with key queries** — it indexes the entire document as a unit. You need either an expression index on that specific key, or a GIN index for general JSONB querying.

**Option 1: Expression index (best for one or two specific keys)**

```sql
CREATE INDEX idx_foo_env ON foo ((config->>'env'));
```

The double parens are required — they tell Postgres this is an expression. Now your query uses the index:

```sql
EXPLAIN SELECT * FROM foo WHERE config->>'env' = 'prod';
-- Index Scan using idx_foo_env on foo
```

Tiny, fast, but only works for `config->>'env'`. You need a separate index for each key you want to query.

**Option 2: GIN index (best for many keys or containment queries)**

```sql
CREATE INDEX idx_foo_config ON foo USING gin (config);
```

Then rewrite the query to use the containment operator:

```sql
SELECT * FROM foo WHERE config @> '{"env": "prod"}';
```

GIN supports a wide range of JSONB operators (`?`, `?|`, `?&`, `@>`, `@?`, `@@`), so one index handles many query shapes. Tradeoffs: bigger on disk, slower writes, slightly slower lookups than the expression index.

**Use `jsonb_path_ops` for smaller GIN indexes** if you only need containment:
```sql
CREATE INDEX idx_foo_config ON foo USING gin (config jsonb_path_ops);
```
About 1/3 smaller, faster lookups, but only supports `@>` (no `?`, `?|`, etc.).

**Rule of thumb:**
- Querying 1-2 specific keys → expression index per key
- Querying many keys or building search → GIN
- Storing structured data you query a lot → consider just making it a real column"""
))

EXAMPLES.append(example(
    user="""How big should my Postgres connection pool be? I have a Node app with PM2 running 8 workers, and I keep hitting 'too many connections' errors.""",
    think="""The user is hitting `FATAL: too many connections` because total connections across all workers exceed Postgres's `max_connections` (default 100, often 100-300 in managed services).

Common mistake: setting pool size = max_connections. Wrong, because you have 8 workers, each with their own pool. If each pool is 20, total is 160 connections.

Math: `total_connections = workers × pool_size_per_worker × replicas`. Need `total_connections < max_connections - reserved_for_admin`.

The right pool size per worker depends on workload, but conventional wisdom is small — like 5-10 connections per worker, not 20. Postgres connections are heavyweight (a forked process per connection, ~10MB of RAM). You don't gain throughput by having 200 connections vs 50.

For high-throughput apps, use a connection pooler like PgBouncer in front of Postgres. The app talks to PgBouncer (hundreds of "connections"), PgBouncer multiplexes onto a small pool of real Postgres connections.

Also: web workers spend most of their time waiting on the network, not actively querying. So max active queries is much less than max connections. Pool size 5-10 per worker is fine for most apps.""",
    answer="""**Math first:** `total_pg_connections = num_workers × pool_size_per_worker × num_replicas`. With 8 PM2 workers and pool size 20, you're at 160 connections per replica — over the default `max_connections = 100`.

**Fix 1: shrink the per-worker pool**

Postgres connections are heavyweight (a forked process, ~10MB RAM each). You don't gain throughput from huge pools — most web workers spend time waiting on the network, not actively querying.

```js
// Node + node-postgres
const pool = new Pool({
  max: 10,         // 10 per worker × 8 workers = 80 total
  min: 2,
  idleTimeoutMillis: 30000,
});
```

For most web apps, 5-10 connections per worker is plenty. Bench it and watch `pg_stat_activity` to see how many are actually active at peak.

**Fix 2: use PgBouncer in front of Postgres**

For high-throughput or many-worker setups, put a connection pooler between your app and Postgres:

```
App workers (160 conns) → PgBouncer → Postgres (20 conns)
```

PgBouncer multiplexes — it accepts hundreds of "connections" from app workers but only maintains a small pool of real Postgres backends. Use `transaction` pooling mode for most web apps:

```ini
# pgbouncer.ini
[databases]
mydb = host=localhost port=5432

[pgbouncer]
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 25
```

**Caveats with `transaction` pool mode**: prepared statements and `SET LOCAL` don't work the way you expect (each transaction can land on a different backend). Use `session` mode if you rely on those.

**Check current usage:**
```sql
SELECT count(*), state FROM pg_stat_activity GROUP BY state;
```

Look for many `idle` connections — those are pool slots you're not using.

**Rule of thumb:** start small (5-10 per worker), add PgBouncer if you outgrow it, and never set pool size based on "max users" or "max RPS" — use `pg_stat_activity` to see what's actually active."""
))


# ── DevOps / Containers / K8s ─────────────────────────────────────

EXAMPLES.append(example(
    user="""My Docker build takes 8 minutes because every code change reinstalls all the npm packages. Here's my Dockerfile:
```dockerfile
FROM node:20
WORKDIR /app
COPY . .
RUN npm install
CMD ["node", "server.js"]
```
What am I doing wrong?""",
    think="""Classic layer-cache mistake. `COPY . .` copies everything including source files, so any source change invalidates the cached layer. The next layer (`npm install`) then has to re-run because its preceding layer changed.

Fix: copy only `package.json` and `package-lock.json` first, run `npm install`, THEN copy the rest. The `npm install` layer is now keyed only on `package.json` content — source changes don't invalidate it.

Other improvements while we're here:
- Use `npm ci` not `npm install` in builds (deterministic, faster, requires lockfile)
- Pin to a specific node version (`node:20.11.0-slim`)
- Use a non-root user
- Multi-stage build to leave dev dependencies behind""",
    answer="""**Layer cache is being invalidated by `COPY . .` before `npm install`.** Every source change copies new files, busting the cache for the install step.

**Fix: copy package files first, install, then copy source:**

```dockerfile
FROM node:20.11.0-slim
WORKDIR /app

# These two layers are cached unless package*.json changes
COPY package.json package-lock.json ./
RUN npm ci --only=production

# Source code change only invalidates from this layer down
COPY . .

USER node
CMD ["node", "server.js"]
```

Now `npm ci` only re-runs when `package.json` or `package-lock.json` change. Source-only edits skip the install entirely — your build drops from 8 minutes to seconds.

**Also worth doing:**

- **`npm ci` not `npm install`** in CI builds — it's deterministic (uses the lockfile exactly), fails on lockfile mismatch, and is faster than `npm install`
- **Pin the base image** — `node:20` follows the tag; `node:20.11.0-slim` is reproducible
- **Use `-slim` or `-alpine`** for smaller images (200MB vs 1GB)
- **`USER node`** — the official Node images include a `node` user; use it instead of running as root
- **`.dockerignore`** to skip `node_modules`, `.git`, tests, README, etc. — these don't belong in your image and slow the COPY down

**`.dockerignore` example:**
```
node_modules
.git
.gitignore
.env*
*.md
tests
coverage
.vscode
```

**Rule of thumb:** order Dockerfile instructions from least-frequently-changed to most-frequently-changed. Base image → system packages → dependencies → source code. Each layer is cached independently."""
))

EXAMPLES.append(example(
    user="""My Docker image is 1.2GB but the actual app is just a Go binary. How do I get it down?""",
    think="""1.2GB is way too big for a Go binary. The user is probably using `golang:1.x` as the runtime base, which includes the entire Go toolchain (700+ MB on its own).

Fix: multi-stage build. Build in `golang:1.x`, copy just the compiled binary into a minimal runtime stage.

Options for the runtime stage:
- `scratch` (~0 MB base) — works for static Go binaries with no dependencies
- `alpine:3.19` (~5 MB) — works if you need a shell or musl libc
- `gcr.io/distroless/static-debian12` (~2 MB) — Google's distroless, has CA certs and tzdata, no shell

For Go specifically, `CGO_ENABLED=0` produces a static binary that runs on `scratch`. If you use sqlite or libc functions, you need to link statically or use distroless.

Also: the binary itself can be smaller with `-ldflags "-s -w"` (strip debug info).""",
    answer="""**Multi-stage build with a `scratch` or distroless runtime stage** — 1.2 GB → ~10-20 MB.

```dockerfile
# ─── Build stage ───────────────────────────────
FROM golang:1.22-alpine AS build

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download

COPY . .
# CGO_ENABLED=0 → static binary, no libc dependency
# -ldflags "-s -w" → strip debug info, ~30% smaller
RUN CGO_ENABLED=0 GOOS=linux go build \
    -ldflags="-s -w" \
    -o /app ./cmd/server

# ─── Runtime stage ─────────────────────────────
FROM scratch
COPY --from=build /app /app

# If your app makes HTTPS calls, you need CA certs
COPY --from=build /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# If your app uses time zones
COPY --from=build /usr/share/zoneinfo /usr/share/zoneinfo

EXPOSE 8080
ENTRYPOINT ["/app"]
```

**The two stages:**
1. **Build stage**: full Go toolchain, builds the binary
2. **Runtime stage**: starts from `scratch` (literally empty), copies just the binary

The final image is ONLY the binary + CA certs + tzdata. No shell, no package manager, no Go toolchain. Final size: ~15 MB for a typical service.

**If you need a shell or musl libc, use `alpine` instead of `scratch`:**
```dockerfile
FROM alpine:3.19
RUN apk add --no-cache ca-certificates tzdata
COPY --from=build /app /app
USER nobody
ENTRYPOINT ["/app"]
```
~5 MB base + your binary.

**If you want distroless** (no shell, but with CA certs and tzdata):
```dockerfile
FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /app /app
ENTRYPOINT ["/app"]
```
~2 MB base, runs as non-root by default.

**Why `scratch` isn't always right:**

- **Need to debug a production container?** No shell, no `kubectl exec` interactivity. Distroless has the same problem.
- **Need libc for cgo or sqlite?** `scratch` won't work — use alpine or distroless.
- **Need DNS resolution from a non-static binary?** You'll hit "no nsswitch.conf" issues — use alpine.

For most pure-Go services, `scratch` is fine. For "I might need to shell in", alpine.

**Rule of thumb:** for compiled languages (Go, Rust, C++), the runtime stage should never be the build base. Strip everything you don't run."""
))

EXAMPLES.append(example(
    user="""K8s liveness vs readiness probes — when do I use which? I have a service that takes 30 seconds to warm up its in-memory cache before it can serve requests.""",
    think="""Common confusion. Both probes check pod health but trigger different actions:

- **Liveness**: "is this pod alive?" Failure → restart the pod. Use for detecting deadlocks, infinite loops, hung processes.
- **Readiness**: "is this pod ready to serve traffic?" Failure → remove from service endpoints (no restart). Use for warmup, dependency health, draining.
- **Startup** (k8s 1.16+): "is this pod still starting up?" Suspends liveness/readiness until startup completes. Use for slow-starting apps.

The user's case (30s warmup) is exactly what startup probes are for. Without one, they'd need to set `initialDelaySeconds: 35` on liveness which delays detection of real problems.

The right setup for the user:
- Startup probe: gives the pod up to N attempts to come online
- Readiness probe: returns 200 once cache is warm and dependencies are healthy
- Liveness probe: returns 200 unless the process is wedged

Common mistake: making liveness probe check downstream dependencies. That causes cascading restarts when a downstream is down. Liveness should only check whether this process can answer requests at all.""",
    answer="""**Three different probes, three different behaviors:**

| Probe | Question | Failure action | When to use |
|---|---|---|---|
| **Startup** | Has the pod finished starting? | Wait longer; eventually fail | Slow-starting apps (warmup, migrations) |
| **Liveness** | Is this pod alive? | **Restart the pod** | Detect deadlocks, hung processes |
| **Readiness** | Should we route traffic to it? | Remove from service endpoints (no restart) | Dependency health, warmup, draining |

**For your 30-second warmup case, use a startup probe + a readiness probe:**

```yaml
spec:
  containers:
  - name: api
    # Startup probe: gives the pod up to 60s to come online
    startupProbe:
      httpGet:
        path: /health/startup
        port: 8080
      failureThreshold: 30
      periodSeconds: 2
      # 30 attempts × 2s = 60s max startup time

    # Readiness probe: only runs after startup succeeds
    readinessProbe:
      httpGet:
        path: /health/ready
        port: 8080
      periodSeconds: 5
      failureThreshold: 2

    # Liveness probe: only runs after startup succeeds
    livenessProbe:
      httpGet:
        path: /health/live
        port: 8080
      periodSeconds: 10
      failureThreshold: 3
```

**What each endpoint should check:**

```python
# /health/startup — has the cache loaded?
@app.get("/health/startup")
def startup():
    if not cache.is_warmed():
        return {"status": "starting"}, 503
    return {"status": "ready"}, 200

# /health/ready — are we ready for traffic?
@app.get("/health/ready")
def ready():
    if not cache.is_warmed():
        return {"status": "warming"}, 503
    if not db.ping():
        return {"status": "db_down"}, 503
    return {"status": "ready"}, 200

# /health/live — is the process responsive?
@app.get("/health/live")
def live():
    # JUST returns 200. Don't check dependencies here.
    return {"status": "alive"}, 200
```

**The critical distinction:**

- **Liveness checks must NOT depend on downstream services.** If your liveness probe checks the database, and the database goes down, k8s restarts every pod in a loop — making outage recovery impossible. Liveness is "can this process answer HTTP requests?"
- **Readiness CAN check dependencies.** If your DB is down, you correctly want traffic to stop hitting this pod. K8s removes you from the service endpoints; once the DB recovers, you go back in rotation. No restart needed.
- **Startup probes prevent both the readiness AND liveness probes from running** until the app is up. Without one, you'd have to set high `initialDelaySeconds` on liveness, which masks real failures during normal operation.

**Common mistakes to avoid:**

- **Liveness probe checking the DB** — cascading restarts during outage
- **Same endpoint for liveness and readiness** — the loop above
- **No startup probe + slow startup** — pod gets restarted before it finishes loading
- **Probe interval too aggressive** — wastes CPU on health checks; 5-10s is usually enough
- **`failureThreshold: 1`** — single transient failure restarts the pod; use 2-3

**Rule of thumb:** liveness = "the process is alive", readiness = "I should get traffic", startup = "give me time to come up". They're independent and you usually want all three."""
))

EXAMPLES.append(example(
    user="""My pod keeps getting OOMKilled even though it's only using ~400MB. I set `requests: 256Mi` and `limits: 512Mi`. What's going on?""",
    think="""OOMKilled at ~400MB with a 512MB limit suggests one of:

1. The pod is briefly spiking above 512MB (RSS measurement is point-in-time, OOM killer uses peak)
2. Something is allocating a buffer that exceeds limits even briefly
3. JVM/Node/Python heap settings don't know about the cgroup limit
4. The kernel page cache or other accounting is hitting the limit

Most likely: the runtime doesn't know about the cgroup limit and is doing its own memory management based on host RAM. JVM is the classic case (sees host RAM, sets heap to 1/4 of host = much more than the limit). Node.js is similar with `--max-old-space-size`. Python doesn't have this issue but native libraries can.

Other thing: requests=256, limits=512. The pod gets scheduled on a node that has 256Mi free. If it actually needs 400, that's fine while running but burst-prone. Should probably bump request closer to limit, or set request=limit (Guaranteed QoS).

Also: container memory limit includes EVERYTHING in the cgroup — heap, stack, native, page cache for files opened by the process. Java/Node devs often forget this.""",
    answer="""**Most likely: your runtime doesn't know about the cgroup memory limit and is sizing itself based on host RAM.** Classic case for JVM, Node.js, and any GC'd language. The "400MB" you see in monitoring is RSS at the moment you measured — actual peak is over 512MB.

**Diagnose first** — get the OOMKill details:
```bash
kubectl describe pod my-pod
# Look for: "Last State: Terminated, Reason: OOMKilled, Exit Code: 137"

# Also check container memory history if you have metrics-server
kubectl top pod my-pod --containers
```

If you have Prometheus, plot `container_memory_working_set_bytes` over time — you'll likely see a spike right before the kill.

**Fix by runtime:**

**JVM** — tell the JVM to respect cgroup limits (default in OpenJDK 10+, but explicit is safer):

```yaml
env:
- name: JAVA_TOOL_OPTIONS
  value: "-XX:MaxRAMPercentage=75.0 -XX:InitialRAMPercentage=50.0"
```

This tells the JVM to use 75% of the cgroup limit for max heap (so the rest covers metaspace, threads, native, etc.). On a 512MB limit, max heap is 384MB.

**Node.js** — set `--max-old-space-size` based on the cgroup limit:

```yaml
env:
- name: NODE_OPTIONS
  value: "--max-old-space-size=384"  # 75% of 512Mi limit
```

Without this, Node uses ~1.5GB by default — way over a 512MB pod limit.

**Python** — usually fine, but watch for:
- Native libraries (numpy, pandas, scientific stack) that allocate outside the GC heap
- Large file reads that get mmap'd into your address space
- Unbounded list/dict growth (generator vs list comprehension)

**Bump the limit if your app legitimately needs more:**

```yaml
resources:
  requests:
    memory: 512Mi
  limits:
    memory: 768Mi  # bumped from 512Mi
```

**Set request close to limit for stability:**

`requests: 256, limits: 512` makes the pod "Burstable" QoS — k8s schedules it expecting 256MB, but the pod can use 512MB if available. If the node is under memory pressure, this pod gets killed first.

For predictable workloads, set `request == limit`:

```yaml
resources:
  requests:
    memory: 512Mi
    cpu: 500m
  limits:
    memory: 512Mi
    cpu: 500m
```

This makes the pod "Guaranteed" QoS — it gets the highest scheduling priority and is killed last under pressure. Tradeoff: less efficient bin-packing on the node.

**Don't forget about the kernel page cache:**

Container memory limit (cgroup v1, by default) counts page cache for files the process has opened. If your app reads a 200MB log file, that 200MB of cache counts against your limit. Cgroup v2 distinguishes "anon" from "file" memory and behaves differently. If you're on a recent k8s version (1.25+), you're likely on cgroup v2.

**Rule of thumb:** GC'd languages need explicit memory hints to respect cgroup limits. Set request close to limit for stable workloads. Don't trust point-in-time monitoring for OOM diagnosis — use peak metrics from your time-series database."""
))

EXAMPLES.append(example(
    user="""ConfigMap or Secret? I have database connection strings, API keys, and feature flags. Where does each go?""",
    think="""ConfigMap = non-sensitive config, plain text in etcd, RBAC controlled
Secret = sensitive data, base64-encoded (NOT encrypted by default!) in etcd, also RBAC controlled

The user's data:
- DB connection strings — contain passwords → Secret
- API keys — sensitive → Secret
- Feature flags — non-sensitive → ConfigMap

Important nuances:
- Secrets are NOT encrypted by default in etcd. They're base64 encoded, which is just obfuscation. Real encryption requires `EncryptionConfiguration` on the API server.
- Both can be mounted as files OR injected as env vars
- File mounts auto-update when the resource changes (~ 1 min lag); env vars don't (require pod restart)
- Use a real secrets manager (External Secrets Operator + AWS SM / Vault) for production — keeps secrets out of git, supports rotation

Common mistakes:
- Putting database URLs in ConfigMap "because they're not really secret"
- Storing secrets as base64 thinking it's encrypted
- Committing Secret YAML to git unencrypted (use Sealed Secrets or SOPS for that)""",
    answer="""**Splitting your data:**

| Data | Where | Why |
|---|---|---|
| Database connection string (with password) | **Secret** | Contains password |
| API keys for third-party services | **Secret** | Sensitive credentials |
| Feature flags | **ConfigMap** | Non-sensitive config |
| Database name, host, port (no password) | **ConfigMap** | Non-sensitive |
| Log level, region, app name | **ConfigMap** | Non-sensitive |

**ConfigMap example:**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: api-config
data:
  LOG_LEVEL: "info"
  FEATURE_NEW_UI: "true"
  REGION: "us-east-1"
  DATABASE_HOST: "db.internal"
  DATABASE_PORT: "5432"
```

**Secret example:**

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: api-secrets
type: Opaque
stringData:
  DATABASE_URL: "postgres://api_user:s3cr3t@db.internal:5432/myapp"
  STRIPE_SECRET_KEY: "sk_live_..."
  GITHUB_TOKEN: "ghp_..."
```

(`stringData` lets you write plain strings; k8s base64-encodes them on apply. The alternative `data` field requires you to base64 yourself, which is annoying.)

**Mount them in your pod:**

```yaml
spec:
  containers:
  - name: api
    image: myapp:v1
    envFrom:
    - configMapRef:
        name: api-config
    - secretRef:
        name: api-secrets
```

`envFrom` injects ALL keys from the ConfigMap/Secret as env vars. Or use `env:` for individual keys.

**Critical thing most people miss: Secrets are NOT encrypted by default.**

A k8s "Secret" is just base64-encoded data in etcd. Anyone with `get secrets` permission (or etcd access) reads them in plaintext. To actually encrypt at rest, you need to configure `EncryptionConfiguration` on the API server:

```yaml
# /etc/kubernetes/encryption-config.yaml
apiVersion: apiserver.config.k8s.io/v1
kind: EncryptionConfiguration
resources:
- resources:
  - secrets
  providers:
  - aescbc:
      keys:
      - name: key1
        secret: <base64-encoded-32-byte-key>
  - identity: {}
```

Most managed k8s services (EKS, GKE, AKS) handle this transparently. Self-hosted clusters need to set it up.

**For production, don't manage secrets in YAML at all.** Use one of:

- **External Secrets Operator** + AWS Secrets Manager / GCP Secret Manager / HashiCorp Vault — pulls secrets from the source of truth into k8s on a schedule
- **Sealed Secrets** — encrypts secrets with the cluster's public key, safe to commit to git
- **SOPS** with age/PGP — encrypts YAML files at rest, decrypts at apply time

**Auto-update behavior:**

- **Mount as file** → updates automatically (~1 min lag) when the ConfigMap/Secret changes. Your app needs to watch the file or be restarted.
- **Inject as env var** → does NOT update on change. Pod must be restarted (`kubectl rollout restart`).

For feature flags you want to toggle without restart, mount as a file and have your app re-read it. For DB credentials, inject as env var and accept that rotation requires a rollout.

**Common mistakes:**

- **Putting DB URLs in ConfigMap** because "the password isn't THAT sensitive" — no, secrets manager
- **Committing Secret YAML to git** unencrypted — use Sealed Secrets or SOPS
- **Sharing one Secret across all environments** — separate dev/staging/prod
- **Granting `get secrets` to too many service accounts** — RBAC is your friend

**Rule of thumb:** if it has a credential, it's a Secret. If it's just config, it's a ConfigMap. For real production, use an external secrets manager and pull into k8s — don't store secrets in YAML files."""
))

EXAMPLES.append(example(
    user="""How do I do zero-downtime deploys in K8s? I'm seeing brief 502s during rollouts.""",
    think="""502s during rollout means traffic is hitting pods that are still starting up (not ready) or pods that are shutting down (already removed from rotation but still receiving traffic). Several things to fix:

1. **Readiness probe** — must return 200 before pod gets traffic. Without it, k8s assumes "ready" as soon as the container starts.
2. **PreStop hook + grace period** — when k8s starts terminating a pod, it (a) removes from service endpoints, (b) sends SIGTERM. If your app exits immediately on SIGTERM, in-flight requests die. Need to drain.
3. **Rolling update strategy** — `maxSurge` and `maxUnavailable` control how many pods can be down at once.
4. **MinReadySeconds** — wait N seconds after a pod becomes ready before considering it actually serving (gives load balancers time to update).
5. **PodDisruptionBudget** — guarantees at least N replicas remain during voluntary disruptions (node drains, etc.).

The 502 specifically usually means: traffic hits a pod that's already shutting down. The fix is the preStop hook: sleep for a few seconds before SIGTERM, so the pod has time to be removed from service endpoints before it stops accepting connections.""",
    answer="""**502s during rollout = traffic hitting pods that are shutting down faster than they're being removed from service endpoints.** Fix is a combination of preStop hook + grace period + readiness probe.

**The full configuration:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1            # at most 1 extra pod during rollout
      maxUnavailable: 0      # never go below desired replicas
  minReadySeconds: 10        # wait 10s after ready before considering "running"
  template:
    spec:
      terminationGracePeriodSeconds: 60
      containers:
      - name: api
        image: myapp:v123
        ports:
        - containerPort: 8080

        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8080
          periodSeconds: 5
          failureThreshold: 2

        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 15"]
```

**Why each piece matters:**

**`maxUnavailable: 0`** — k8s never takes a pod offline before its replacement is ready. With `maxSurge: 1`, the rollout briefly runs 4 pods (3 old + 1 new), waits for the new one to be ready, then takes one old one down. No downtime.

**`readinessProbe`** — k8s only routes traffic to pods that pass readiness. If your app takes time to warm up, this prevents 502s during pod startup.

**`minReadySeconds: 10`** — even after a pod is "ready", wait 10s before counting it. This gives external load balancers (ALB, GCLB) time to see the pod as healthy in their own probes. Without this, k8s declares the pod ready and routes traffic, but the cloud LB hasn't picked it up yet.

**`preStop` hook with sleep + `terminationGracePeriodSeconds`** — this is the critical fix for the 502s. Here's the timeline:

```
t=0:   k8s decides to terminate pod
       k8s removes pod from service endpoints (async, ~1s lag)
       k8s runs preStop hook → sleep 15
t=15:  preStop completes, k8s sends SIGTERM
       app starts graceful shutdown (drain connections, finish in-flight requests)
t=15+: app finishes shutdown OR graceperiod expires
       k8s sends SIGKILL if app hasn't exited
t=60:  termination grace period max
```

The `sleep 15` gives the service endpoints update time to propagate before SIGTERM hits the app. New connections stop arriving during the sleep, so when SIGTERM fires, only in-flight requests need to drain.

**Your app must handle SIGTERM gracefully**:

```python
# Python with FastAPI / uvicorn
import signal
import asyncio

shutdown_event = asyncio.Event()

def handle_sigterm(signum, frame):
    shutdown_event.set()

signal.signal(signal.SIGTERM, handle_sigterm)

# In your shutdown handler:
@app.on_event("shutdown")
async def shutdown():
    # Stop accepting new requests
    # Wait for in-flight requests to finish
    # Close DB connections
    await db.close()
```

For Node.js Express:

```js
const server = app.listen(8080);
process.on('SIGTERM', () => {
  console.log('SIGTERM received, draining...');
  server.close(() => {
    console.log('drained, exiting');
    process.exit(0);
  });
});
```

**PodDisruptionBudget** for voluntary disruptions (node drains, cluster upgrades):

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: api-pdb
spec:
  minAvailable: 2  # never go below 2 pods even during node drain
  selector:
    matchLabels:
      app: api
```

This prevents `kubectl drain` from removing all your pods at once.

**Verify with a load test during rollout:**

```bash
# Continuously hit the service
while true; do curl -s -o /dev/null -w "%{http_code}\n" https://api.example.com/health; sleep 0.1; done

# In another terminal, trigger a rollout
kubectl set image deployment/api api=myapp:v124
```

You should see only 200s, no 502s.

**Rule of thumb:** zero-downtime deploys need 5 things — readiness probe, `maxUnavailable: 0`, preStop hook with sleep, graceful SIGTERM handling, and `minReadySeconds` for cloud LB convergence. Missing any one of them causes intermittent errors during rollouts."""
))

EXAMPLES.append(example(
    user="""I want to run my container as non-root with a read-only root filesystem. What does the Dockerfile + k8s manifest look like?""",
    think="""Two parts: Dockerfile creates a non-root user and prepares writable directories, k8s manifest enforces non-root + read-only-root + drops capabilities.

Dockerfile:
- Create a user (uid > 1000 to avoid clashing with system users)
- Pre-create any directories the app needs to write to
- Set USER directive

K8s manifest:
- securityContext.runAsNonRoot: true (refuses to run if image is root)
- securityContext.runAsUser: <uid>
- securityContext.readOnlyRootFilesystem: true
- emptyDir volumes for /tmp and other writable paths
- securityContext.allowPrivilegeEscalation: false
- securityContext.capabilities.drop: [ALL]

Also seccompProfile and AppArmor profiles for extra hardening.

The user just asked for non-root + read-only-root. Cover that thoroughly with the rest as "while you're here".""",
    answer="""**Both layers matter — the Dockerfile creates the user and prepares writable dirs, the k8s manifest enforces the constraints at runtime.**

**Dockerfile:**

```dockerfile
FROM node:20.11.0-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build

# ─── Runtime ───────────────────────────────
FROM node:20.11.0-slim

# Create non-root user (uid 1001 to avoid conflicting with system users)
RUN groupadd -r app -g 1001 && \
    useradd -r -u 1001 -g app -d /home/app -s /sbin/nologin app && \
    mkdir -p /home/app && \
    chown -R app:app /home/app

WORKDIR /app
COPY --from=build --chown=app:app /app/dist ./dist
COPY --from=build --chown=app:app /app/node_modules ./node_modules
COPY --from=build --chown=app:app /app/package.json ./

USER app

EXPOSE 8080
CMD ["node", "dist/server.js"]
```

The `--chown=app:app` on COPY ensures files are owned by the new user. Without it, they'd be owned by root and your app couldn't read them.

**Kubernetes manifest:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
spec:
  template:
    spec:
      securityContext:
        runAsNonRoot: true     # refuses to start if image runs as root
        runAsUser: 1001
        runAsGroup: 1001
        fsGroup: 1001
        seccompProfile:
          type: RuntimeDefault

      containers:
      - name: api
        image: myapp:v1
        securityContext:
          readOnlyRootFilesystem: true
          allowPrivilegeEscalation: false
          capabilities:
            drop:
            - ALL

        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: cache
          mountPath: /home/app/.cache
        - name: npm
          mountPath: /home/app/.npm

      volumes:
      - name: tmp
        emptyDir: {}
      - name: cache
        emptyDir: {}
      - name: npm
        emptyDir: {}
```

**Why each setting:**

- **`runAsNonRoot: true`** — k8s refuses to start the container if the image's USER is root (or unset). This catches Dockerfile mistakes at deploy time, not runtime.
- **`runAsUser: 1001`** — explicit uid override, even if the image USER is different.
- **`fsGroup: 1001`** — files in mounted volumes get this group, so the app can read them.
- **`readOnlyRootFilesystem: true`** — the container's root filesystem is read-only. Any writes (logs, temp files, caches) must go to a mounted volume. This stops attackers from writing webshells or modifying binaries.
- **`emptyDir` for writable dirs** — `/tmp`, app cache directories, anything your app legitimately needs to write to. `emptyDir` is per-pod ephemeral storage.
- **`allowPrivilegeEscalation: false`** — blocks setuid binaries from gaining privileges.
- **`capabilities: drop: [ALL]`** — drops all Linux capabilities. Most apps don't need any. Add specific ones back if needed (e.g., `NET_BIND_SERVICE` if you need to bind to port < 1024).
- **`seccompProfile: RuntimeDefault`** — restricts the syscalls the container can make to a safe default set. Catches a wide class of kernel exploits.

**Test it locally first:**

```bash
docker run --rm \
    --user 1001:1001 \
    --read-only \
    --tmpfs /tmp \
    --tmpfs /home/app/.cache \
    --cap-drop ALL \
    myapp:v1
```

If the container fails to start, find what it's trying to write to and add an `emptyDir` mount.

**Common issues you'll hit:**

- **`EACCES` writing to /app** — your app is trying to write to its working directory. Either don't, or mount an `emptyDir` over `/app/data`.
- **`EROFS` errors during npm/pip operations** — you have node_modules or site-packages in the read-only root. Build them in the image, don't install at runtime.
- **`/tmp not writable`** — add an emptyDir at `/tmp`.
- **Logs not appearing** — your app is writing to a file. Stop doing that — log to stdout, k8s collects it.
- **Image works locally but fails in k8s** — `runAsNonRoot: true` catches it. Set USER in your Dockerfile.

**Rule of thumb:** non-root + read-only-root + drop capabilities is the baseline for production containers. Builds add ~30 minutes the first time, then it's free for every subsequent service."""
))

EXAMPLES.append(example(
    user="""I'm setting up CI/CD and the team is debating "build once, deploy many" vs "build per environment". What's the right approach?""",
    think=""""Build once, deploy many" is the standard answer for any modern delivery pipeline:
- Build a single artifact (container image, jar, binary) once
- Tag with the commit SHA and a semver
- Promote that EXACT artifact through environments (dev → staging → prod)
- Configuration differences come from env vars / config files / secrets per environment

The alternative (rebuild for each environment) means:
- "Works in staging" doesn't actually mean it'll work in prod (different builds!)
- You can't compare what's deployed where — different SHAs of the "same" version
- Reproducibility is harder — what you tested isn't what shipped
- More CI time, more cost

The user is in a debate so they need ammunition. Cover the "why", show the structure, address the natural objection ("but our env-specific configs are baked in!").""",
    answer="""**"Build once, deploy many" is the right answer.** The same artifact (image, binary, etc.) is built ONE time and promoted through environments unchanged. Configuration differences are injected at deploy time, not baked in at build time.

**Why "build per environment" is wrong:**

- **What you tested isn't what ships.** A staging build with `NODE_ENV=staging` baked in is a different binary than a prod build with `NODE_ENV=production` baked in. Bugs can hide in the differences.
- **You can't reproduce what's in prod.** Your prod deploy is the result of a CI run that won't recreate exactly the same artifact (timestamps, transitive dependency drift, etc.).
- **No cross-environment comparison.** Staging is on commit `abc123 (staging build)` and prod is on `abc123 (prod build)` — they're not the same thing.
- **More CI time.** Building 3x means 3x cost.

**The right pipeline structure:**

```
┌─ Commit ─┐
│   ↓      │
│  Build   │  ← happens ONCE
│   ↓      │
│  Test    │
│   ↓      │
│  Tag with SHA + version
│   ↓      │
│  Push to registry: myapp:abc123
│          │
└──────┬───┘
       │
       ├──→ Deploy to dev    (config: dev env vars + dev secrets)
       │
       ├──→ Deploy to staging (config: staging env vars + staging secrets)
       │      ↑ tests pass
       │      ↓ manual approval
       └──→ Deploy to prod   (config: prod env vars + prod secrets)
```

**Implementation in GitHub Actions:**

```yaml
# .github/workflows/build.yml
on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    outputs:
      image_tag: ${{ steps.tag.outputs.tag }}
    steps:
    - uses: actions/checkout@v4
    - id: tag
      run: echo "tag=${GITHUB_SHA::7}" >> $GITHUB_OUTPUT

    - name: Build and push
      uses: docker/build-push-action@v5
      with:
        push: true
        tags: |
          myapp:${{ steps.tag.outputs.tag }}
          myapp:latest

  deploy-dev:
    needs: build
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - run: |
        kubectl set image deployment/api api=myapp:${{ needs.build.outputs.image_tag }} -n dev
        kubectl rollout status deployment/api -n dev

  deploy-staging:
    needs: deploy-dev
    runs-on: ubuntu-latest
    environment: staging
    steps:
    - run: |
        kubectl set image deployment/api api=myapp:${{ needs.build.outputs.image_tag }} -n staging
        kubectl rollout status deployment/api -n staging

  deploy-prod:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production  # GitHub environment with required reviewers
    steps:
    - run: |
        kubectl set image deployment/api api=myapp:${{ needs.build.outputs.image_tag }} -n prod
        kubectl rollout status deployment/api -n prod
```

**Notice:** the SAME image tag (`abc1234`) is deployed to all three environments. The build job runs ONCE.

**"But our environments need different config!"**

Right — that's what env vars and ConfigMaps/Secrets are for. The image stays the same; the config differs:

```yaml
# k8s/overlays/prod/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
- ../../base
patches:
- patch: |-
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: api
    spec:
      template:
        spec:
          containers:
          - name: api
            env:
            - name: ENV
              value: "production"
            - name: LOG_LEVEL
              value: "info"
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: prod-db
                  key: url
```

Same image, different env vars per environment. The app reads `process.env.ENV` and behaves accordingly.

**For TRULY different builds (e.g., different feature flags compiled in):**

Use **runtime feature flags** (LaunchDarkly, Unleash, your own DB-backed flag service) instead of build-time flags. The image is identical; the flag service tells each environment what to enable.

If you genuinely cannot avoid build-time differences (e.g., JS bundle with `process.env.PUBLIC_API_URL` baked in by webpack), then build once and substitute at deploy time:

```dockerfile
# In nginx-served frontend, replace placeholder at container start
ENV API_URL=__API_URL__
ENTRYPOINT ["/docker-entrypoint.sh"]
```

```bash
#!/bin/sh
# docker-entrypoint.sh
sed -i "s|__API_URL__|$API_URL|g" /usr/share/nginx/html/main.js
nginx -g 'daemon off;'
```

Now the image is built once with placeholders, and the placeholder is substituted from env var at container start.

**Rule of thumb:** the artifact you test should be the artifact you ship. Build once, tag with SHA, promote unchanged through environments, inject config at deploy time."""
))


# ── Web / API Design ──────────────────────────────────────────────

EXAMPLES.append(example(
    user="""I'm building a payments API. How do I make POST requests safe to retry? My client times out sometimes and I'm worried about double-charging.""",
    think="""Idempotency keys. Client generates a unique key per logical operation, sends it as a header. Server stores `(key → response)` for some window (24h is standard). If the same key arrives again, return the stored response without re-executing.

Critical detail: the key must be tied to the request body. If the client retries with the same key but a different body, that's a programming error and the server should return 422.

Storage: Redis with TTL is perfect. The stored value is the full response (status + body).

Stripe and other payment APIs use `Idempotency-Key` header. Length: random UUID v4 is fine. Server validates length and char set.

Race condition: two simultaneous requests with the same key. Use SET NX to atomically claim the key, return "in flight" 409 if already claimed but no result yet, or have the second request poll until the first completes.""",
    answer="""**Idempotency keys** — the client generates a unique key per logical operation, the server caches the response and returns it on retry.

**Client side:**
```js
const idempotencyKey = crypto.randomUUID();

async function chargeWithRetry(amount) {
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      return await fetch("/api/charges", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Idempotency-Key": idempotencyKey,  // SAME key on retry
        },
        body: JSON.stringify({ amount }),
      });
    } catch (e) {
      if (attempt === 2) throw e;
      await sleep(2 ** attempt * 1000);
    }
  }
}
```

The same `idempotencyKey` on every retry — that's the whole point. A different key = a different operation.

**Server side:**

```python
import hashlib
import json
import redis

r = redis.Redis()

@app.post("/api/charges")
def create_charge(req: ChargeRequest, idempotency_key: str = Header(alias="Idempotency-Key")):
    if not idempotency_key:
        raise HTTPException(400, "Idempotency-Key header required")

    # Hash the body so we can detect "same key, different body" misuse
    body_hash = hashlib.sha256(json.dumps(req.dict(), sort_keys=True).encode()).hexdigest()
    cache_key = f"idem:{idempotency_key}"

    # Atomic claim: set cache_key to "in_progress" only if it doesn't exist
    claimed = r.set(cache_key, json.dumps({"status": "in_progress", "body_hash": body_hash}),
                     nx=True, ex=86400)

    if not claimed:
        # Key already exists — either result is ready or another request is in flight
        existing = json.loads(r.get(cache_key))
        if existing["body_hash"] != body_hash:
            raise HTTPException(422, "Idempotency-Key reused with different request body")
        if existing.get("status") == "in_progress":
            raise HTTPException(409, "Request still processing — retry in a moment")
        # Return cached response
        return existing["response"]

    # First request with this key — actually do the work
    try:
        charge = stripe.charge(amount=req.amount, source=req.source)
        response = {"id": charge.id, "status": "succeeded", "amount": req.amount}
        r.set(cache_key, json.dumps({
            "status": "complete",
            "body_hash": body_hash,
            "response": response,
        }), ex=86400)
        return response
    except Exception as e:
        r.delete(cache_key)  # let client retry
        raise
```

**Key details:**

- **Hash the body** — if the client reuses a key with a different body, that's a bug and you should reject with 422
- **24-hour TTL** is standard; longer = more storage, shorter = legitimate retries fail
- **Atomic claim** with `SET NX` prevents two simultaneous requests from both executing
- **Return 409 for in-flight** — tells the client to wait and retry, don't double-execute
- **Delete on failure** — let the client retry with the same key
- **Store the FULL response**, not just the resource ID — the client needs the same response on retry

**Rule of thumb:** any non-GET endpoint that could be retried (network timeouts, mobile flakiness, queue reprocessing) needs idempotency keys. GET requests are naturally idempotent."""
))

EXAMPLES.append(example(
    user="""My SPA is getting CORS errors when I call my API. I added Access-Control-Allow-Origin: * and now it works for GET but POST with cookies fails. What gives?""",
    think="""Two problems:
1. `Access-Control-Allow-Origin: *` is incompatible with `Access-Control-Allow-Credentials: true`. The wildcard isn't allowed when sending cookies — must be a specific origin.
2. POST with credentials triggers a CORS preflight (OPTIONS request). The server has to respond to OPTIONS with the right CORS headers BEFORE the browser sends the actual POST.

Fix:
- Set `Access-Control-Allow-Origin` to the specific origin (or echo the Origin header from a whitelist)
- Set `Access-Control-Allow-Credentials: true`
- Handle OPTIONS preflight explicitly (or use a CORS middleware)
- Frontend must set `credentials: "include"` on fetch
- Allowed methods/headers must be in the preflight response

Common gotchas:
- `Vary: Origin` header so caches don't serve the wrong CORS response
- Custom headers like `Authorization` or `X-CSRF-Token` need to be in `Access-Control-Allow-Headers`""",
    answer="""**Two problems compounding:** wildcard origin doesn't work with credentials, and POST with credentials triggers a preflight your server isn't handling.

**Problem 1: `*` is incompatible with credentials**

Browsers reject `Access-Control-Allow-Origin: *` when the request sends cookies. You must echo a specific origin:

```python
# FastAPI with middleware
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://myapp.com", "https://www.myapp.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-CSRF-Token"],
    expose_headers=["X-Request-Id"],
    max_age=3600,  # cache the preflight for 1 hour
)
```

**Problem 2: preflight (OPTIONS) handling**

POST with `Content-Type: application/json` is a "non-simple" request, so the browser sends an OPTIONS request first asking "can I do this?" Your server has to respond with the CORS headers — the actual POST never happens until preflight passes.

The `CORSMiddleware` above handles this automatically. If you're rolling your own:

```python
@app.options("/api/{path:path}")
def cors_preflight(path: str, origin: str = Header(default="")):
    if origin not in ALLOWED_ORIGINS:
        raise HTTPException(403)
    return Response(headers={
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Credentials": "true",
        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization, X-CSRF-Token",
        "Access-Control-Max-Age": "3600",
        "Vary": "Origin",
    })
```

**Frontend must explicitly opt into sending credentials:**

```js
fetch("https://api.myapp.com/charges", {
  method: "POST",
  credentials: "include",   // ← must be set
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ amount: 100 }),
});
```

Without `credentials: "include"`, the browser doesn't send the cookie even if the CORS headers allow it.

**Other gotchas:**

- **`Vary: Origin` header** — without this, a CDN/proxy might cache the response for one origin and serve it to another (wrong CORS headers)
- **Custom headers must be allowlisted** — `Authorization`, `X-CSRF-Token`, anything custom must appear in `Access-Control-Allow-Headers` in the preflight response
- **Preflights are cached** — `Access-Control-Max-Age` controls how long. Use 1 hour for stable APIs, less if you're iterating
- **Subdomains count as different origins** — `myapp.com` and `www.myapp.com` need separate entries
- **Don't echo arbitrary `Origin` headers** — validate against an allowlist, otherwise you're effectively allow-all-origins

**Rule of thumb:** when CORS breaks, check (1) is the origin specific (not `*`)?, (2) does the preflight respond with all the right headers?, (3) does the client set `credentials: "include"`?"""
))

EXAMPLES.append(example(
    user="""How should I version my REST API? URL versioning, header versioning, or content negotiation?""",
    think="""All three work and have different tradeoffs:

1. **URL versioning** (`/v1/users`, `/v2/users`): most common, easy to debug, easy to route, easy to document. Some say it's "not RESTful" but pragmatically it's the winner.

2. **Header versioning** (`Accept-Version: 2`): cleaner URLs, but harder to debug (can't paste URL in browser), harder for new users to discover.

3. **Content negotiation** (`Accept: application/vnd.myapp.v2+json`): "proper" REST, but ugly and hard to test.

Most successful APIs (Stripe, GitHub, Twilio) use URL versioning OR header versioning. Stripe specifically uses an `Stripe-Version` header AND defaults based on account creation date.

For most teams, URL versioning is the right answer because:
- Easy to test in browser/curl
- Easy to route different versions to different services
- Easy for users to understand
- Easy to deprecate (sunset old version's URLs)

The harder question is "when do you create v2?" and "how long do you support v1?" — that's a policy question, not a technical one.""",
    answer="""**URL versioning (`/v1/users`, `/v2/users`) is the pragmatic winner for most teams.** It's "less pure" than content negotiation but wins on every practical axis: easy to debug, easy to route, easy to document, easy to deprecate.

**The three options:**

| Approach | Example | Pros | Cons |
|---|---|---|---|
| URL | `GET /v1/users` | Easy debug, easy route, easy docs | "Not RESTful" (cosmetic) |
| Header | `Accept-Version: 2` | Cleaner URLs | Harder debug, harder discover |
| Content negotiation | `Accept: application/vnd.api.v2+json` | "Proper REST" | Ugly, hard to test |

**Use URL versioning** unless you have a specific reason not to:

```
GET https://api.myapp.com/v1/users/123
GET https://api.myapp.com/v2/users/123
```

In code:
```python
# FastAPI
from fastapi import APIRouter

v1 = APIRouter(prefix="/v1")
v2 = APIRouter(prefix="/v2")

@v1.get("/users/{user_id}")
def get_user_v1(user_id: int):
    return {"id": user_id, "name": "Alice"}

@v2.get("/users/{user_id}")
def get_user_v2(user_id: int):
    return {"id": user_id, "name": "Alice", "email": "alice@example.com", "created_at": "..."}

app.include_router(v1)
app.include_router(v2)
```

**When to bump the version:**

- **Breaking change to response shape** (removed field, renamed field, type change)
- **Breaking change to request shape** (now-required field, removed field)
- **Breaking change to behavior** (auth flow change, status code change for the same condition)

**NOT a version bump:**

- Adding a new optional field to the response (clients should ignore unknown fields)
- Adding a new endpoint
- Adding a new optional query parameter
- Performance improvements
- Bug fixes that align with documented behavior

**Deprecation policy** (this is the hard part):

```
v1 supported:    indefinitely (2-year minimum from v2 release)
v1 deprecated:   warning header + docs banner (12 months before sunset)
v1 sunset:       returns 410 Gone with link to migration guide
```

Add a deprecation header to v1 responses:
```
Sunset: Sat, 31 Dec 2027 23:59:59 GMT
Deprecation: true
Link: <https://docs.myapp.com/migrate-v1-to-v2>; rel="deprecation"
```

These are standard headers (RFC 8594, RFC draft) that some HTTP clients understand.

**Don't go past v3 if you can avoid it.** Each version is a maintenance burden (bug fixes, security patches, docs). If you find yourself at v5, you're either doing major redesigns too often or your "version bumps" should have been new endpoints.

**Stripe's hybrid approach** (if you want to copy a great example):
- Default version baked into the customer's account (locked at signup time)
- `Stripe-Version` header to override per request
- Sunset old versions slowly (6-12 month warning, then 12+ months still working)
- Detailed changelog with migration paths

**Rule of thumb:** URL versioning, bump only on breaking changes, support old versions for 12-24 months, communicate sunset dates aggressively in docs, headers, and email."""
))

EXAMPLES.append(example(
    user="""I need to rate-limit my API. What's the difference between token bucket, leaky bucket, and fixed window?""",
    think="""Three classic rate limiting algorithms:

1. **Fixed window**: count requests in a fixed time window (e.g., per-minute counter that resets at :00). Simple but has the "boundary spike" problem — a client can do 2x the limit in a 2-second window straddling the boundary.

2. **Sliding window**: smooth out the boundary spike. Either log every request and count those in the last N seconds (memory-intensive) or use a weighted average of the current and previous window.

3. **Token bucket**: bucket holds N tokens, refills at rate R per second, each request takes 1 token. Allows burstiness up to N. Most common in practice.

4. **Leaky bucket**: requests enter a queue, are processed at fixed rate. Smooths out traffic, no burstiness.

Token bucket is what most APIs use because it allows reasonable bursts while enforcing an average rate. Leaky bucket is for when you absolutely need smooth output (e.g., calling an upstream that has a hard rate limit).

Implementation: Redis with atomic Lua script for the token bucket (atomic check-and-update of tokens + last-refill time).""",
    answer="""**Token bucket** is what most APIs should use — allows bursts up to a configured ceiling while enforcing an average rate. The others are for specific use cases.

**Quick comparison:**

| Algorithm | Behavior | Best for |
|---|---|---|
| Fixed window | Counter resets every N seconds | Simple internal limits |
| Sliding window | Smoothed count over rolling window | Strict per-window enforcement |
| Token bucket | N tokens, refills at R/sec, each req takes 1 | Most APIs (allows reasonable bursts) |
| Leaky bucket | Fixed-rate queue draining | Smoothing traffic to a strict downstream |

**Token bucket implementation** (Redis + Lua for atomicity):

```python
import time
import redis

r = redis.Redis()

# Lua script: atomically refill tokens based on elapsed time, take 1 token if available
TOKEN_BUCKET_LUA = '''
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens = tonumber(bucket[1]) or capacity
local last_refill = tonumber(bucket[2]) or now

-- Refill tokens based on elapsed time
local elapsed = math.max(0, now - last_refill)
tokens = math.min(capacity, tokens + elapsed * refill_rate)

if tokens >= 1 then
    tokens = tokens - 1
    redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
    redis.call('EXPIRE', key, 3600)
    return 1  -- allowed
else
    redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
    redis.call('EXPIRE', key, 3600)
    return 0  -- rate limited
end
'''

bucket_script = r.register_script(TOKEN_BUCKET_LUA)

def allow_request(client_id: str, capacity: int = 100, refill_per_sec: float = 10) -> bool:
    # Token bucket: 100 token capacity, refills at 10/sec → 10 RPS sustained, burst to 100
    return bucket_script(
        keys=[f"ratelimit:{client_id}"],
        args=[capacity, refill_per_sec, time.time()]
    ) == 1
```

The Lua script is atomic — refill calculation and token consumption happen in one Redis operation, no race condition between simultaneous requests.

**FastAPI middleware:**

```python
from fastapi import HTTPException, Request

@app.middleware("http")
async def rate_limit(request: Request, call_next):
    client_id = request.headers.get("X-API-Key") or request.client.host

    if not allow_request(client_id, capacity=100, refill_per_sec=10):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={
                "Retry-After": "10",
                "X-RateLimit-Limit": "100",
                "X-RateLimit-Remaining": "0",
            }
        )
    return await call_next(request)
```

**Tuning parameters:**

- `capacity` = max burst size (100 means a client can fire 100 requests instantly)
- `refill_per_sec` = sustained rate (10 means 10 requests per second average)
- Adjust based on your endpoints — read-heavy can tolerate more, write-heavy needs less

**Per-key strategy:**

- **Per IP** (anonymous endpoints): catches abusers, has false positives behind NAT
- **Per API key** (authenticated): much more accurate, lets you give different tiers different limits
- **Per user + per IP combined**: most flexible, more state to manage

**Returning rate limit info to clients:**

```http
HTTP/1.1 200 OK
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1633024800

HTTP/1.1 429 Too Many Requests
Retry-After: 10
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1633024810
```

The standard headers (`X-RateLimit-*`) tell well-behaved clients when they can retry without trial-and-error.

**Rule of thumb:** token bucket for general APIs, leaky bucket for queueing to a strict downstream, fixed window for simple "max N per minute per IP" filtering. Always return `Retry-After` on 429 so clients can back off correctly."""
))

EXAMPLES.append(example(
    user="""How do I securely verify webhook signatures? My payment provider is sending HMAC signatures and I want to make sure I'm validating them correctly.""",
    think="""Webhook signature verification is one of those things that's easy to get subtly wrong. Common mistakes:

1. **Using `==` for comparison** — vulnerable to timing attacks. Use `hmac.compare_digest()`.
2. **Not including the timestamp** in the signed payload — replay attacks
3. **Computing HMAC over the parsed body** instead of the raw bytes — JSON serialization differences (key order, whitespace) make signatures fail
4. **Using the wrong hash algorithm** — Stripe uses SHA-256, others vary
5. **Forgetting to check the timestamp window** — old signatures still validate forever

Stripe pattern (most common): `Stripe-Signature: t=1234,v1=abcdef...` Hash is HMAC-SHA256 of `{timestamp}.{body}` with the webhook secret as key.

Reject if:
- Timestamp is more than 5 minutes old
- HMAC doesn't match (timing-safe comparison)""",
    answer="""**Three things must all be right**: hash the RAW request body (not parsed), include a timestamp in the signed payload, compare with `hmac.compare_digest` (timing-safe).

**The standard pattern (Stripe-style):**

```python
import hmac
import hashlib
import time
from fastapi import Request, HTTPException

WEBHOOK_SECRET = os.environ["WEBHOOK_SECRET"]
TOLERANCE_SECONDS = 300  # 5 minutes

@app.post("/webhooks/payments")
async def webhook(request: Request):
    # 1. Read the RAW body bytes — don't let any framework parse it
    raw_body = await request.body()

    # 2. Get the signature header
    sig_header = request.headers.get("X-Webhook-Signature")
    if not sig_header:
        raise HTTPException(401, "missing signature")

    # 3. Parse: "t=1234567890,v1=abcdef..."
    try:
        parts = dict(p.split("=", 1) for p in sig_header.split(","))
        timestamp = int(parts["t"])
        signature = parts["v1"]
    except (ValueError, KeyError):
        raise HTTPException(401, "malformed signature header")

    # 4. Reject old signatures (replay protection)
    if abs(time.time() - timestamp) > TOLERANCE_SECONDS:
        raise HTTPException(401, "signature timestamp too old or in the future")

    # 5. Compute expected signature: HMAC-SHA256({timestamp}.{raw_body})
    signed_payload = f"{timestamp}.".encode() + raw_body
    expected = hmac.new(
        WEBHOOK_SECRET.encode(),
        signed_payload,
        hashlib.sha256
    ).hexdigest()

    # 6. Timing-safe comparison
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(401, "signature mismatch")

    # NOW we can trust the body and parse it
    event = json.loads(raw_body)
    process_webhook(event)
    return {"received": True}
```

**Why each step matters:**

**Hash the RAW body bytes:**
```python
raw_body = await request.body()  # bytes
# NOT this:
# event = await request.json()  # parsed dict — order/whitespace differs
```
JSON parsing and re-serializing changes whitespace, key order, and number formatting. Your HMAC won't match. **Always hash the bytes the sender sent.**

**Include the timestamp in the signed payload:**

If you only sign the body, an attacker who captures one valid webhook can replay it forever. The timestamp is included in the HMAC input AND checked against current time, so old signatures expire automatically.

**Use `hmac.compare_digest`, not `==`:**

```python
# WRONG — leaks information via timing
if expected == signature:
    ...

# RIGHT — constant time comparison
if hmac.compare_digest(expected, signature):
    ...
```

`==` short-circuits on the first differing byte, so an attacker can guess the signature one byte at a time by measuring response time. `compare_digest` always compares all bytes regardless.

**Common framework gotchas:**

**FastAPI/Starlette**: be careful with middleware that might consume the body. Calling `request.body()` after another middleware has read the stream returns empty bytes. Use a custom route class or read the body in the handler before any other framework code touches it.

**Express (Node.js)**:
```js
// MUST use express.raw, not express.json, on the webhook route
app.post('/webhooks',
  express.raw({ type: 'application/json' }),
  (req, res) => {
    const sig = req.headers['x-webhook-signature'];
    const expected = crypto.createHmac('sha256', WEBHOOK_SECRET)
      .update(`${timestamp}.`)
      .update(req.body)  // raw Buffer
      .digest('hex');

    if (!crypto.timingSafeEqual(Buffer.from(expected), Buffer.from(sig))) {
      return res.status(401).send();
    }
    // Now parse the body yourself
    const event = JSON.parse(req.body.toString());
    res.json({ received: true });
  }
);
```

**Use the SDK if your provider has one.** Stripe's `stripe.webhooks.constructEvent(body, sig, secret)` does all of this correctly. So does GitHub's, Slack's, etc. Don't reimplement what the provider already gives you.

**Rule of thumb:** raw bytes + timestamp + timing-safe compare. Never parse before verifying. Reject signatures older than 5 minutes."""
))

EXAMPLES.append(example(
    user="""What's the right format for API error responses? I want something machine-readable that includes enough info for debugging.""",
    think="""RFC 7807 "Problem Details for HTTP APIs" is the standard. JSON object with required `type` (URI identifying the problem type) and `title`, plus optional `status`, `detail`, `instance`, and any extension fields.

Example:
```json
{
  "type": "https://api.example.com/errors/insufficient-balance",
  "title": "Insufficient balance",
  "status": 402,
  "detail": "Account balance of $5.00 is below the required $10.00",
  "instance": "/accounts/12345/transactions/abc",
  "balance": 5.00,
  "required": 10.00
}
```

Benefits:
- Standard, well-known format
- Machine-readable (clients can switch on `type`)
- Human-readable (`title`, `detail`)
- Extensible (add custom fields)
- Includes `instance` URI for tracing

Many APIs do their own custom format. RFC 7807 is increasingly the default for new APIs because it's standardized.""",
    answer="""**Use RFC 7807 "Problem Details for HTTP APIs"** — a standard JSON format with required `type` and `title` plus optional fields. Machine-readable, human-readable, extensible.

**Standard fields:**

```json
{
  "type": "https://api.example.com/errors/insufficient-balance",
  "title": "Insufficient balance",
  "status": 402,
  "detail": "Account balance of $5.00 is below the required $10.00 for this transaction",
  "instance": "/accounts/12345/transactions/abc",
  "balance": 5.00,
  "required": 10.00,
  "request_id": "req_abc123"
}
```

| Field | Required | Purpose |
|---|---|---|
| `type` | Yes | URI identifying the problem type — clients switch on this |
| `title` | Yes | Short human-readable summary |
| `status` | No | HTTP status code (matches the response) |
| `detail` | No | Specific explanation for THIS occurrence |
| `instance` | No | URI of the specific resource that errored |
| (custom) | No | Any additional fields specific to this error type |

**`type` is the critical field.** It's a stable URI that identifies the error class. Clients should `if error.type == "https://api.example.com/errors/insufficient-balance"` rather than parsing the title or detail (which can change). The URI doesn't have to resolve, but it's nice if it points to docs.

**Implementation in FastAPI:**

```python
from fastapi import HTTPException
from fastapi.responses import JSONResponse

class ProblemDetail(HTTPException):
    def __init__(self, type_: str, title: str, status: int, detail: str = None, **extras):
        self.type_ = type_
        self.title = title
        self.detail = detail
        self.extras = extras
        super().__init__(status_code=status, detail=detail)

@app.exception_handler(ProblemDetail)
async def problem_handler(request, exc: ProblemDetail):
    body = {
        "type": exc.type_,
        "title": exc.title,
        "status": exc.status_code,
    }
    if exc.detail:
        body["detail"] = exc.detail
    body["instance"] = str(request.url.path)
    body["request_id"] = request.headers.get("X-Request-Id", "")
    body.update(exc.extras)

    return JSONResponse(
        status_code=exc.status_code,
        content=body,
        media_type="application/problem+json",  # standardized content type
    )

# Usage in handlers:
@app.post("/transactions")
def create_transaction(req: TransactionRequest):
    account = get_account(req.account_id)
    if account.balance < req.amount:
        raise ProblemDetail(
            type_="https://api.example.com/errors/insufficient-balance",
            title="Insufficient balance",
            status=402,
            detail=f"Account balance of ${account.balance} is below the required ${req.amount}",
            balance=float(account.balance),
            required=float(req.amount),
        )
    # ...
```

**`Content-Type: application/problem+json`** is the standardized media type — set it on every error response. Clients can use it to detect "this is an error response" without checking the status code.

**Validation errors (multiple problems in one response):**

For form/body validation, use a custom extension field:

```json
{
  "type": "https://api.example.com/errors/validation",
  "title": "Validation failed",
  "status": 422,
  "detail": "The request body has 2 validation errors",
  "instance": "/users",
  "errors": [
    {"field": "email", "error": "must be a valid email address"},
    {"field": "age", "error": "must be at least 18"}
  ]
}
```

**Common error type URIs to define:**

```
https://api.example.com/errors/validation        → 422
https://api.example.com/errors/not-found         → 404
https://api.example.com/errors/forbidden         → 403
https://api.example.com/errors/conflict          → 409
https://api.example.com/errors/rate-limited      → 429
https://api.example.com/errors/internal          → 500
```

You can also use IETF-defined types from `https://iana.org/assignments/http-problem-types/`.

**What NOT to do:**

```json
// DON'T return varying shapes
{"error": "something failed"}
{"err_msg": "something else"}
{"errors": ["a", "b"], "msg": "..."}

// DON'T leak internal details
{"error": "Database connection failed: postgresql://prod-db:5432"}
{"error": "Stack trace: Traceback (most recent call last)..."}

// DON'T mix success and error fields
{"success": false, "error": "...", "data": null}
```

Pick one consistent shape (RFC 7807) and use it everywhere.

**Rule of thumb:** RFC 7807 with `application/problem+json`, stable `type` URIs that clients can switch on, custom extension fields for context, never leak internal details (stack traces, DB errors, file paths)."""
))


def main() -> None:
    """Append all examples to the target file."""
    if not TARGET.exists():
        raise SystemExit(f"Target file not found: {TARGET}")
    with TARGET.open("a", encoding="utf-8") as f:
        for ex in EXAMPLES:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Appended {len(EXAMPLES)} examples to {TARGET}")


if __name__ == "__main__":
    main()
