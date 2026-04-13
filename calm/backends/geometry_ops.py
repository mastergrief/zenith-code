"""
CALM Geometry backend — area, perimeter, volume, distance.

Models botch formulas for non-trivial shapes. Pure math.
"""

from __future__ import annotations

import math


def circle_area(radius: float) -> float:
    """Area of a circle."""
    return math.pi * float(radius) ** 2


def circle_circumference(radius: float) -> float:
    """Circumference of a circle."""
    return 2 * math.pi * float(radius)


def sphere_volume(radius: float) -> float:
    """Volume of a sphere."""
    return (4 / 3) * math.pi * float(radius) ** 3


def sphere_surface_area(radius: float) -> float:
    """Surface area of a sphere."""
    return 4 * math.pi * float(radius) ** 2


def cylinder_volume(radius: float, height: float) -> float:
    """Volume of a cylinder."""
    return math.pi * float(radius) ** 2 * float(height)


def cylinder_surface_area(radius: float, height: float) -> float:
    """Surface area of a cylinder (including top and bottom)."""
    r, h = float(radius), float(height)
    return 2 * math.pi * r * (r + h)


def cone_volume(radius: float, height: float) -> float:
    """Volume of a cone."""
    return (1 / 3) * math.pi * float(radius) ** 2 * float(height)


def cone_surface_area(radius: float, slant_height: float) -> float:
    """Surface area of a cone (including base). slant_height, not vertical height."""
    r, s = float(radius), float(slant_height)
    return math.pi * r * (r + s)


def triangle_area(base: float, height: float) -> float:
    """Area of a triangle (base × height / 2)."""
    return float(base) * float(height) / 2


def triangle_area_heron(a: float, b: float, c: float) -> float:
    """Area of a triangle using Heron's formula (3 sides)."""
    a, b, c = float(a), float(b), float(c)
    s = (a + b + c) / 2
    val = s * (s - a) * (s - b) * (s - c)
    return math.sqrt(val) if val >= 0 else -1.0


def trapezoid_area(a: float, b: float, height: float) -> float:
    """Area of a trapezoid (parallel sides a, b and height)."""
    return (float(a) + float(b)) * float(height) / 2


def ellipse_area(a: float, b: float) -> float:
    """Area of an ellipse (semi-major a, semi-minor b)."""
    return math.pi * float(a) * float(b)


def rectangle_diagonal(width: float, height: float) -> float:
    """Diagonal of a rectangle."""
    return math.sqrt(float(width) ** 2 + float(height) ** 2)


def distance_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two 2D points."""
    return math.sqrt((float(x2) - float(x1)) ** 2 + (float(y2) - float(y1)) ** 2)


def distance_3d(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> float:
    """Euclidean distance between two 3D points."""
    return math.sqrt((float(x2)-float(x1))**2 + (float(y2)-float(y1))**2 + (float(z2)-float(z1))**2)


def degrees_to_radians(deg: float) -> float:
    """Convert degrees to radians."""
    return math.radians(float(deg))


def radians_to_degrees(rad: float) -> float:
    """Convert radians to degrees."""
    return math.degrees(float(rad))


def polygon_interior_angle(sides: int) -> float:
    """Interior angle of a regular polygon."""
    n = int(sides)
    if n < 3:
        return -1.0
    return (n - 2) * 180 / n


def hypotenuse(a: float, b: float) -> float:
    """Hypotenuse of a right triangle (Pythagorean theorem)."""
    return math.sqrt(float(a) ** 2 + float(b) ** 2)


GEOMETRY_NL_PATTERNS = [
    (r'area\s+of\s+(?:a\s+)?circle\s+(?:with\s+)?radius\s+([\d.]+)', 'circle_area({0})'),
    (r'circumference\s+of\s+(?:a\s+)?circle\s+(?:with\s+)?radius\s+([\d.]+)', 'circle_circumference({0})'),
    (r'volume\s+of\s+(?:a\s+)?sphere\s+(?:with\s+)?radius\s+([\d.]+)', 'sphere_volume({0})'),
    (r'surface\s+area\s+of\s+(?:a\s+)?sphere\s+(?:with\s+)?radius\s+([\d.]+)', 'sphere_surface_area({0})'),
    (r'volume\s+of\s+(?:a\s+)?cylinder\s+(?:with\s+)?radius\s+([\d.]+)\s+(?:and\s+)?height\s+([\d.]+)', 'cylinder_volume({0}, {1})'),
    (r'volume\s+of\s+(?:a\s+)?cone\s+(?:with\s+)?radius\s+([\d.]+)\s+(?:and\s+)?height\s+([\d.]+)', 'cone_volume({0}, {1})'),
    (r'area\s+of\s+(?:a\s+)?triangle\s+(?:with\s+)?base\s+([\d.]+)\s+(?:and\s+)?height\s+([\d.]+)', 'triangle_area({0}, {1})'),
    (r'area\s+of\s+(?:a\s+)?trapezoid\s+.*?(?:sides?\s+)?([\d.]+)\s+and\s+([\d.]+)\s+.*?height\s+([\d.]+)', 'trapezoid_area({0}, {1}, {2})'),
    (r'hypotenuse.*?([\d.]+)\s+and\s+([\d.]+)', 'hypotenuse({0}, {1})'),
    (r'distance\s+(?:between|from)\s+\(?([\d.]+)\s*,\s*([\d.]+)\)?\s+(?:and|to)\s+\(?([\d.]+)\s*,\s*([\d.]+)\)?', 'distance_2d({0}, {1}, {2}, {3})'),
    (r'(?:interior|internal)\s+angle\s+of\s+(?:a\s+)?regular\s+triangle', 'polygon_interior_angle(3)'),
    (r'(?:interior|internal)\s+angle\s+of\s+(?:a\s+)?regular\s+square', 'polygon_interior_angle(4)'),
    (r'(?:interior|internal)\s+angle\s+of\s+(?:a\s+)?regular\s+pentagon', 'polygon_interior_angle(5)'),
    (r'(?:interior|internal)\s+angle\s+of\s+(?:a\s+)?regular\s+hexagon', 'polygon_interior_angle(6)'),
    (r'(?:interior|internal)\s+angle\s+of\s+(?:a\s+)?regular\s+heptagon', 'polygon_interior_angle(7)'),
    (r'(?:interior|internal)\s+angle\s+of\s+(?:a\s+)?regular\s+octagon', 'polygon_interior_angle(8)'),
    (r'(?:interior|internal)\s+angle\s+of\s+(?:a\s+)?regular\s+(\d+).(?:sided|gon)', 'polygon_interior_angle({0})'),
]

GEOMETRY_FUNCTIONS = {
    "circle_area": circle_area,
    "circle_circumference": circle_circumference,
    "sphere_volume": sphere_volume,
    "sphere_surface_area": sphere_surface_area,
    "cylinder_volume": cylinder_volume,
    "cylinder_surface_area": cylinder_surface_area,
    "cone_volume": cone_volume,
    "cone_surface_area": cone_surface_area,
    "triangle_area": triangle_area,
    "triangle_area_heron": triangle_area_heron,
    "trapezoid_area": trapezoid_area,
    "ellipse_area": ellipse_area,
    "rectangle_diagonal": rectangle_diagonal,
    "distance_2d": distance_2d,
    "distance_3d": distance_3d,
    "degrees_to_radians": degrees_to_radians,
    "radians_to_degrees": radians_to_degrees,
    "polygon_interior_angle": polygon_interior_angle,
    "hypotenuse": hypotenuse,
}
