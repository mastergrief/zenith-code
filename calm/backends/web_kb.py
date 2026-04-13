"""
CALM Web development knowledge backend — HTML elements, CSS concepts, browser APIs.

Models confuse semantic HTML, hallucinate CSS properties, mix up browser APIs.
"""

from __future__ import annotations

_DATA_VERSION = "2025-01"

_HTML_SEMANTIC = {
    "header": {"purpose": "Introductory content or navigation links", "typically": "logo, nav, search", "not": "generic container (use div)"},
    "nav": {"purpose": "Navigation links section", "a11y": "screen readers identify main navigation"},
    "main": {"purpose": "Primary content of the page", "rules": "one per page, not inside header/footer/nav/aside"},
    "article": {"purpose": "Self-contained, independently distributable content", "examples": "blog post, news article, comment", "test": "would it make sense in an RSS feed?"},
    "section": {"purpose": "Thematic grouping of content, typically with heading", "vs_div": "section implies semantic grouping, div is purely structural"},
    "aside": {"purpose": "Tangentially related content", "examples": "sidebar, pull quote, related links"},
    "footer": {"purpose": "Footer for section or page", "typically": "copyright, contact, sitemap links"},
    "figure": {"purpose": "Self-contained content with optional caption (figcaption)", "examples": "image, diagram, code listing"},
    "details": {"purpose": "Expandable disclosure widget", "children": "summary (visible toggle) + content"},
    "dialog": {"purpose": "Modal or non-modal dialog box", "api": ".showModal(), .show(), .close()"},
    "time": {"purpose": "Machine-readable date/time", "example": '<time datetime="2025-01-15">January 15</time>'},
    "mark": {"purpose": "Highlighted/marked text", "use": "search results, key terms"},
    "abbr": {"purpose": "Abbreviation with expansion", "example": '<abbr title="HyperText Markup Language">HTML</abbr>'},
}

_CSS_LAYOUT = {
    "flexbox": {"display": "flex", "direction": "row (default) or column", "properties": {"container": ["justify-content", "align-items", "flex-wrap", "gap", "flex-direction"], "item": ["flex-grow", "flex-shrink", "flex-basis", "align-self", "order"]}, "use_when": "1D layout (row OR column)"},
    "grid": {"display": "grid", "properties": {"container": ["grid-template-columns", "grid-template-rows", "gap", "grid-template-areas", "justify-items", "align-items"], "item": ["grid-column", "grid-row", "grid-area", "justify-self", "align-self"]}, "use_when": "2D layout (rows AND columns)"},
    "position": {"values": {"static": "default, normal flow", "relative": "offset from normal position, creates stacking context", "absolute": "removed from flow, positioned relative to nearest positioned ancestor", "fixed": "removed from flow, positioned relative to viewport", "sticky": "hybrid: relative until scroll threshold, then fixed"}},
    "box-model": {"layers": ["content", "padding", "border", "margin"], "box-sizing": {"content-box": "width = content only (default, confusing)", "border-box": "width = content + padding + border (recommended)"}},
}

_CSS_UNITS = {
    "px": {"type": "absolute", "description": "CSS pixel (device-independent)", "use": "borders, shadows, precise control"},
    "em": {"type": "relative", "relative_to": "parent font-size", "gotcha": "compounds — 1.2em in 1.2em = 1.44x root", "use": "component-relative sizing"},
    "rem": {"type": "relative", "relative_to": "root (:root/html) font-size", "use": "consistent spacing, font sizes"},
    "%": {"type": "relative", "relative_to": "parent element", "use": "widths, responsive layout"},
    "vw": {"type": "relative", "relative_to": "1% of viewport width", "use": "full-width sections, responsive text"},
    "vh": {"type": "relative", "relative_to": "1% of viewport height", "gotcha": "mobile: includes address bar. Use dvh instead."},
    "dvh": {"type": "relative", "relative_to": "1% of dynamic viewport height", "use": "mobile full-screen (accounts for URL bar)"},
    "ch": {"type": "relative", "relative_to": "width of '0' character", "use": "fixed-width content (code, forms)"},
    "fr": {"type": "flexible", "relative_to": "fraction of available space in grid", "use": "grid-template-columns: 1fr 2fr"},
    "clamp()": {"type": "function", "syntax": "clamp(min, preferred, max)", "use": "responsive font-size: clamp(1rem, 2vw, 2rem)"},
}

_BROWSER_STORAGE = {
    "localStorage": {"capacity": "5-10 MB", "persistence": "permanent (until cleared)", "scope": "per origin", "api": "getItem/setItem/removeItem", "sync": True, "use": "user preferences, cached data"},
    "sessionStorage": {"capacity": "5-10 MB", "persistence": "tab lifetime", "scope": "per origin + per tab", "api": "getItem/setItem/removeItem", "sync": True, "use": "form state, single-session data"},
    "cookies": {"capacity": "4 KB per cookie", "persistence": "configurable (expires/max-age)", "scope": "per domain + path", "sent_with": "every HTTP request (Set-Cookie/Cookie headers)", "security": ["HttpOnly", "Secure", "SameSite"], "use": "auth tokens, server-readable state"},
    "IndexedDB": {"capacity": "large (100s MB+)", "persistence": "permanent", "api": "async, transactional, key-value", "use": "offline apps, large datasets, structured data"},
    "Cache API": {"capacity": "large", "persistence": "permanent", "api": "caches.open/match/put", "use": "service worker caching, offline-first PWAs"},
}


def html_element(name: str) -> dict:
    """Get semantic HTML element details."""
    key = str(name).lower().strip().lstrip('<').rstrip('>')
    entry = _HTML_SEMANTIC.get(key)
    if not entry:
        return {"error": f"Unknown: {name}", "valid": sorted(_HTML_SEMANTIC.keys())}
    return {"element": f"<{key}>", **entry}


def css_layout(system: str) -> dict:
    """Get CSS layout system details (flexbox, grid, position, box-model)."""
    key = str(system).lower().strip()
    for k, v in _CSS_LAYOUT.items():
        if key in k or k in key:
            return {"layout": k, **v}
    return {"error": f"Unknown: {system}", "valid": list(_CSS_LAYOUT.keys())}


def css_unit(unit: str) -> dict:
    """Get CSS unit details."""
    key = str(unit).lower().strip()
    entry = _CSS_UNITS.get(key)
    if not entry:
        for k, v in _CSS_UNITS.items():
            if key in k:
                return {"unit": k, **v}
        return {"error": f"Unknown: {unit}", "valid": list(_CSS_UNITS.keys())}
    return {"unit": key, **entry}


def browser_storage(name: str) -> dict:
    """Get browser storage API details."""
    key = str(name).lower().strip()
    for k, v in _BROWSER_STORAGE.items():
        if key in k.lower():
            return {"storage": k, **v}
    return {"error": f"Unknown: {name}", "valid": list(_BROWSER_STORAGE.keys())}


def flexbox_vs_grid() -> dict:
    """Compare CSS Flexbox vs Grid."""
    return {"flexbox": _CSS_LAYOUT["flexbox"], "grid": _CSS_LAYOUT["grid"],
            "summary": "Flexbox = 1D (row or column). Grid = 2D (rows and columns). Use both."}


def localstorage_vs_cookies() -> dict:
    """Compare localStorage vs cookies."""
    return {"localStorage": _BROWSER_STORAGE["localStorage"], "cookies": _BROWSER_STORAGE["cookies"],
            "summary": "localStorage: client-only, 5MB, no auto-send. Cookies: sent with requests, 4KB, HttpOnly for security."}


def em_vs_rem() -> dict:
    """Compare em vs rem CSS units."""
    return {"em": _CSS_UNITS["em"], "rem": _CSS_UNITS["rem"],
            "recommendation": "Use rem for consistent sizing. Use em when you want sizes relative to parent."}


WEB_FUNCTIONS = {
    "html_element": html_element,
    "css_layout": css_layout,
    "css_unit": css_unit,
    "browser_storage": browser_storage,
    "flexbox_vs_grid": flexbox_vs_grid,
    "localstorage_vs_cookies": localstorage_vs_cookies,
    "em_vs_rem": em_vs_rem,
}

WEB_NL_PATTERNS = [
    (r'(?:what is|explain|when to use)\s+(?:the\s+)?<?(header|nav|main|article|section|aside|footer|figure|details|dialog|time|mark)>?\s+(?:element|tag)', 'html_element("{0}")'),
    (r'(?:what is|explain)\s+(?:CSS\s+)?(flexbox|grid|position|box.model)', 'css_layout("{0}")'),
    (r'(?:compare|difference|vs)\s+(?:CSS\s+)?flexbox\s+(?:and|vs)\s+grid', 'flexbox_vs_grid()'),
    (r'(?:compare|difference|vs)\s+(?:CSS\s+)?em\s+(?:and|vs)\s+rem', 'em_vs_rem()'),
    (r'(?:what is|explain)\s+(?:the\s+)?(?:CSS\s+)?(px|em|rem|vw|vh|dvh|ch|fr|%)\s+unit', 'css_unit("{0}")'),
    (r'(?:compare|difference|vs)\s+localStorage\s+(?:and|vs)\s+cookies', 'localstorage_vs_cookies()'),
    (r'(?:what is|explain)\s+(localStorage|sessionStorage|IndexedDB|cookies|Cache API)\s+(?:storage|API)?', 'browser_storage("{0}")'),
]
