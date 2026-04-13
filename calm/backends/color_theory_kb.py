"""
CALM Color theory knowledge backend — color models, named colors, relationships.

Models confuse color spaces, hallucinate hex codes for named colors.
"""

from __future__ import annotations

_DATA_VERSION = "2025-01"

_NAMED_COLORS = {
    "red": "#FF0000", "green": "#008000", "blue": "#0000FF",
    "white": "#FFFFFF", "black": "#000000", "yellow": "#FFFF00",
    "cyan": "#00FFFF", "magenta": "#FF00FF", "orange": "#FFA500",
    "purple": "#800080", "pink": "#FFC0CB", "brown": "#A52A2A",
    "gray": "#808080", "silver": "#C0C0C0", "gold": "#FFD700",
    "navy": "#000080", "teal": "#008080", "maroon": "#800000",
    "olive": "#808000", "lime": "#00FF00", "aqua": "#00FFFF",
    "coral": "#FF7F50", "salmon": "#FA8072", "khaki": "#F0E68C",
    "indigo": "#4B0082", "violet": "#EE82EE", "turquoise": "#40E0D0",
    "crimson": "#DC143C", "tomato": "#FF6347", "orchid": "#DA70D6",
}

_COLOR_MODELS = {
    "RGB": {"description": "Red, Green, Blue — additive color model", "range": "0-255 per channel", "use": "screens, digital images", "channels": 3},
    "RGBA": {"description": "RGB with Alpha (transparency) channel", "range": "0-255 per channel, alpha 0-1 or 0-255", "use": "web, compositing"},
    "HSL": {"description": "Hue, Saturation, Lightness — human-intuitive", "range": "H: 0-360°, S: 0-100%, L: 0-100%", "use": "color pickers, CSS"},
    "HSV": {"description": "Hue, Saturation, Value — similar to HSL", "alias": "HSB", "range": "H: 0-360°, S: 0-100%, V: 0-100%", "use": "image editing"},
    "CMYK": {"description": "Cyan, Magenta, Yellow, Key (black) — subtractive", "range": "0-100% per channel", "use": "print"},
    "HEX": {"description": "#RRGGBB hexadecimal representation of RGB", "range": "#000000 to #FFFFFF", "use": "web (CSS, HTML)"},
    "LAB": {"description": "Lightness, a (green-red), b (blue-yellow) — perceptually uniform", "use": "color science, color difference calculations"},
    "Pantone": {"description": "Standardized proprietary spot colors", "use": "brand colors, print production"},
}

_HARMONIES = {
    "complementary": {"description": "Two colors opposite on color wheel (180° apart)", "effect": "high contrast, vibrant", "example": "red + cyan"},
    "analogous": {"description": "Three colors adjacent on color wheel (30° apart)", "effect": "harmonious, low contrast", "example": "red + orange + yellow"},
    "triadic": {"description": "Three colors equally spaced (120° apart)", "effect": "balanced, colorful", "example": "red + green + blue"},
    "split-complementary": {"description": "Base + two colors adjacent to complement", "effect": "contrast with less tension than complementary"},
    "tetradic": {"description": "Four colors forming a rectangle on wheel", "alias": "double complementary", "effect": "rich, requires careful balance"},
    "monochromatic": {"description": "Variations of one hue (different saturation/lightness)", "effect": "clean, elegant", "example": "light blue, medium blue, dark blue"},
}


def named_color(name: str) -> str:
    """Get hex code for a named CSS color."""
    key = str(name).lower().strip()
    return _NAMED_COLORS.get(key, f"unknown: {name}")


def color_model(name: str) -> dict:
    """Get details about a color model."""
    key = str(name).upper().strip()
    entry = _COLOR_MODELS.get(key)
    if not entry:
        return {"error": f"Unknown: {name}", "valid": list(_COLOR_MODELS.keys())}
    return {"model": key, **entry}


def color_harmony(name: str) -> dict:
    """Get details about a color harmony."""
    key = str(name).lower().strip()
    for k, v in _HARMONIES.items():
        if key in k:
            return {"harmony": k, **v}
    return {"error": f"Unknown: {name}", "valid": list(_HARMONIES.keys())}


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple."""
    h = str(hex_color).lstrip('#')
    if len(h) == 3:
        h = ''.join(c * 2 for c in h)
    if len(h) != 6:
        return (-1, -1, -1)
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB to hex color."""
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def complementary_color(hex_color: str) -> str:
    """Get complementary color (opposite on color wheel)."""
    r, g, b = hex_to_rgb(hex_color)
    if r == -1:
        return "invalid color"
    return rgb_to_hex(255 - r, 255 - g, 255 - b)


def lighten(hex_color: str, amount: float = 0.2) -> str:
    """Lighten a color by a percentage (0-1)."""
    r, g, b = hex_to_rgb(hex_color)
    if r == -1:
        return "invalid color"
    a = float(amount)
    r = min(255, int(r + (255 - r) * a))
    g = min(255, int(g + (255 - g) * a))
    b = min(255, int(b + (255 - b) * a))
    return rgb_to_hex(r, g, b)


def darken(hex_color: str, amount: float = 0.2) -> str:
    """Darken a color by a percentage (0-1)."""
    r, g, b = hex_to_rgb(hex_color)
    if r == -1:
        return "invalid color"
    a = float(amount)
    r = max(0, int(r * (1 - a)))
    g = max(0, int(g * (1 - a)))
    b = max(0, int(b * (1 - a)))
    return rgb_to_hex(r, g, b)


def luminance(hex_color: str) -> float:
    """Relative luminance (WCAG formula)."""
    r, g, b = hex_to_rgb(hex_color)
    if r == -1:
        return -1.0
    def linearize(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return round(0.2126 * linearize(r) + 0.7152 * linearize(g) + 0.0722 * linearize(b), 4)


def contrast_ratio(hex1: str, hex2: str) -> float:
    """WCAG contrast ratio between two colors (1:1 to 21:1)."""
    l1 = luminance(hex1)
    l2 = luminance(hex2)
    if l1 < 0 or l2 < 0:
        return -1.0
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return round((lighter + 0.05) / (darker + 0.05), 2)


def wcag_rating(hex_fg: str, hex_bg: str) -> dict:
    """WCAG accessibility rating for text on background."""
    ratio = contrast_ratio(hex_fg, hex_bg)
    return {
        "ratio": ratio,
        "AA_normal": ratio >= 4.5,
        "AA_large": ratio >= 3.0,
        "AAA_normal": ratio >= 7.0,
        "AAA_large": ratio >= 4.5,
    }


def list_named_colors() -> dict:
    """List all named colors with hex codes."""
    return dict(_NAMED_COLORS)


def mix_colors(hex1: str, hex2: str, ratio: float = 0.5) -> str:
    """Mix two colors by ratio (0 = all hex1, 1 = all hex2)."""
    r1, g1, b1 = hex_to_rgb(hex1)
    r2, g2, b2 = hex_to_rgb(hex2)
    if r1 == -1 or r2 == -1:
        return "invalid color"
    t = float(ratio)
    return rgb_to_hex(
        int(r1 + (r2 - r1) * t),
        int(g1 + (g2 - g1) * t),
        int(b1 + (b2 - b1) * t),
    )


def is_dark(hex_color: str) -> bool:
    """Whether a color is perceptually dark (luminance < 0.5)."""
    return luminance(hex_color) < 0.5


def is_light(hex_color: str) -> bool:
    """Whether a color is perceptually light (luminance >= 0.5)."""
    return luminance(hex_color) >= 0.5


def grayscale(hex_color: str) -> str:
    """Convert color to grayscale."""
    r, g, b = hex_to_rgb(hex_color)
    if r == -1:
        return "invalid color"
    gray = int(0.299 * r + 0.587 * g + 0.114 * b)
    return rgb_to_hex(gray, gray, gray)


COLOR_THEORY_FUNCTIONS = {
    "named_color": named_color,
    "color_model": color_model,
    "color_harmony": color_harmony,
    "hex_to_rgb": hex_to_rgb,
    "rgb_to_hex": rgb_to_hex,
    "complementary_color": complementary_color,
    "lighten": lighten,
    "darken": darken,
    "luminance": luminance,
    "contrast_ratio": contrast_ratio,
    "wcag_rating": wcag_rating,
    "list_named_colors": list_named_colors,
    "mix_colors": mix_colors,
    "is_dark": is_dark,
    "is_light": is_light,
    "grayscale": grayscale,
}

COLOR_THEORY_NL_PATTERNS = [
    (r'(?:hex|color code)\s+(?:of|for)\s+(\w+)', 'named_color("{0}")'),
    (r'(?:what is|explain)\s+(RGB|CMYK|HSL|HSV|LAB|HEX)\s+(?:color\s+)?(?:model|space)', 'color_model("{0}")'),
    (r'(?:complementary|opposite)\s+(?:color\s+)?(?:of|to)\s+#?([0-9a-fA-F]{6})', 'complementary_color("#{0}")'),
    (r'(?:WCAG|contrast|accessibility)\s+(?:ratio|rating)\s+(?:of|between)\s+#?([0-9a-fA-F]{6})\s+(?:and|on|vs)\s+#?([0-9a-fA-F]{6})', 'contrast_ratio("#{0}", "#{1}")'),
    (r'(?:lighten|darken)\s+#?([0-9a-fA-F]{6})', None),
    (r'(?:mix|blend)\s+#?([0-9a-fA-F]{6})\s+(?:and|with)\s+#?([0-9a-fA-F]{6})', 'mix_colors("#{0}", "#{1}")'),
]
