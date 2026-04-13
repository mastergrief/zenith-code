"""
CALM Color backend — hex/RGB/HSL conversion, WCAG contrast, named colors.

Models hallucinate color conversions and contrast ratios. Pure math.
"""

from __future__ import annotations

import colorsys
import re

_NAMED_COLORS = {
    "black": (0, 0, 0), "white": (255, 255, 255),
    "red": (255, 0, 0), "green": (0, 128, 0), "blue": (0, 0, 255),
    "yellow": (255, 255, 0), "cyan": (0, 255, 255), "magenta": (255, 0, 255),
    "orange": (255, 165, 0), "purple": (128, 0, 128), "pink": (255, 192, 203),
    "gray": (128, 128, 128), "grey": (128, 128, 128),
    "lime": (0, 255, 0), "navy": (0, 0, 128), "teal": (0, 128, 128),
    "maroon": (128, 0, 0), "olive": (128, 128, 0), "aqua": (0, 255, 255),
    "silver": (192, 192, 192), "coral": (255, 127, 80),
    "salmon": (250, 128, 114), "gold": (255, 215, 0),
    "indigo": (75, 0, 130), "violet": (238, 130, 238),
    "turquoise": (64, 224, 208), "tan": (210, 180, 140),
    "crimson": (220, 20, 60), "khaki": (240, 230, 140),
    "plum": (221, 160, 221), "sienna": (160, 82, 45),
    "tomato": (255, 99, 71), "chocolate": (210, 105, 30),
}


def _parse_color(color: str) -> tuple:
    """Parse hex, rgb(), or named color to (r, g, b)."""
    color = color.strip().lower()
    if color in _NAMED_COLORS:
        return _NAMED_COLORS[color]
    # Hex: #RGB, #RRGGBB
    if color.startswith("#"):
        h = color[1:]
        if len(h) == 3:
            h = h[0]*2 + h[1]*2 + h[2]*2
        if len(h) == 6:
            return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    # rgb(r, g, b)
    m = re.match(r'rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', color)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    raise ValueError(f"can't parse color: {color}")


def color_hex_to_rgb(hex_color: str) -> str:
    """Convert hex color to RGB. Returns 'rgb(r, g, b)'."""
    try:
        r, g, b = _parse_color(hex_color)
        return f"rgb({r}, {g}, {b})"
    except ValueError as e:
        return str(e)


def color_rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB to hex color."""
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def color_rgb_to_hsl(r: int, g: int, b: int) -> str:
    """Convert RGB to HSL. Returns 'hsl(h, s%, l%)'."""
    h, l, s = colorsys.rgb_to_hls(int(r)/255, int(g)/255, int(b)/255)
    return f"hsl({round(h*360)}, {round(s*100)}%, {round(l*100)}%)"


def color_hsl_to_rgb(h: int, s: int, l: int) -> str:
    """Convert HSL to RGB. h=0-360, s=0-100, l=0-100."""
    r, g, b = colorsys.hls_to_rgb(int(h)/360, int(l)/100, int(s)/100)
    return f"rgb({round(r*255)}, {round(g*255)}, {round(b*255)})"


def _relative_luminance(r: int, g: int, b: int) -> float:
    """WCAG 2.1 relative luminance."""
    def linearize(c):
        c = c / 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * linearize(r) + 0.7152 * linearize(g) + 0.0722 * linearize(b)


def color_contrast(color1: str, color2: str) -> str:
    """WCAG 2.1 contrast ratio between two colors. Returns ratio and pass/fail."""
    try:
        r1, g1, b1 = _parse_color(color1)
        r2, g2, b2 = _parse_color(color2)
    except ValueError as e:
        return str(e)
    l1 = _relative_luminance(r1, g1, b1)
    l2 = _relative_luminance(r2, g2, b2)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    ratio = (lighter + 0.05) / (darker + 0.05)
    aa_normal = "PASS" if ratio >= 4.5 else "FAIL"
    aa_large = "PASS" if ratio >= 3.0 else "FAIL"
    aaa_normal = "PASS" if ratio >= 7.0 else "FAIL"
    return f"{ratio:.2f}:1 (AA normal: {aa_normal}, AA large: {aa_large}, AAA: {aaa_normal})"


def color_lighten(color: str, amount: int = 10) -> str:
    """Lighten a color by amount% (0-100). Returns hex."""
    try:
        r, g, b = _parse_color(color)
    except ValueError as e:
        return str(e)
    h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
    l = min(1.0, l + int(amount) / 100)
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return f"#{round(r2*255):02x}{round(g2*255):02x}{round(b2*255):02x}"


def color_darken(color: str, amount: int = 10) -> str:
    """Darken a color by amount% (0-100). Returns hex."""
    try:
        r, g, b = _parse_color(color)
    except ValueError as e:
        return str(e)
    h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
    l = max(0.0, l - int(amount) / 100)
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return f"#{round(r2*255):02x}{round(g2*255):02x}{round(b2*255):02x}"


def color_complementary(color: str) -> str:
    """Complementary color (opposite on color wheel). Returns hex."""
    try:
        r, g, b = _parse_color(color)
    except ValueError as e:
        return str(e)
    h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
    h = (h + 0.5) % 1.0
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return f"#{round(r2*255):02x}{round(g2*255):02x}{round(b2*255):02x}"


def color_name(color: str) -> str:
    """Find the closest named color."""
    try:
        r, g, b = _parse_color(color)
    except ValueError as e:
        return str(e)
    best_name = "unknown"
    best_dist = float("inf")
    for name, (nr, ng, nb) in _NAMED_COLORS.items():
        if name == "grey":
            continue
        d = (r - nr)**2 + (g - ng)**2 + (b - nb)**2
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name


COLOR_NL_PATTERNS = [
    (r"(?:WCAG|contrast).*?([#][0-9a-fA-F]{3,8})\s+(?:and|on|vs|over|against)\s+([#][0-9a-fA-F]{3,8})", 'color_contrast("{0}", "{1}")'),
    (r"convert\s+([#][0-9a-fA-F]{3,8})\s+to\s+(?:RGB|rgb)", 'color_hex_to_rgb("{0}")'),
    (r"complement(?:ary)?\s+(?:color\s+)?(?:of|for)\s+([#\w]+)", 'color_complementary("{0}")'),
    (r"(?:lighten|darken)\s+([#\w]+)\s+(?:by\s+)?(\d+)%?", 'color_lighten("{0}", {1})'),
]

COLOR_FUNCTIONS = {
    "color_hex_to_rgb": color_hex_to_rgb,
    "color_rgb_to_hex": color_rgb_to_hex,
    "color_rgb_to_hsl": color_rgb_to_hsl,
    "color_hsl_to_rgb": color_hsl_to_rgb,
    "color_contrast": color_contrast,
    "color_lighten": color_lighten,
    "color_darken": color_darken,
    "color_complementary": color_complementary,
    "color_name": color_name,
}
