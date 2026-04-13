"""
CALM Coordinate/geospatial backend — lat/long, DMS, haversine, bearings.

Models botch haversine, mess up DMS conversion, confuse lat/long order.
"""

from __future__ import annotations

import math


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points."""
    R = 6371.0  # Earth mean radius km
    la1, lo1 = math.radians(float(lat1)), math.radians(float(lon1))
    la2, lo2 = math.radians(float(lat2)), math.radians(float(lon2))
    dlat = la2 - la1
    dlon = lo2 - lo1
    a = math.sin(dlat / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlon / 2) ** 2
    return round(R * 2 * math.asin(math.sqrt(a)), 2)


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles between two lat/lon points."""
    return round(haversine(lat1, lon1, lat2, lon2) * 0.621371, 2)


def decimal_to_dms(decimal_degrees: float) -> str:
    """Convert decimal degrees to DMS string (e.g. 40°26'46\"N)."""
    dd = float(decimal_degrees)
    d = int(dd)
    m = int((abs(dd) - abs(d)) * 60)
    s = round((abs(dd) - abs(d) - m / 60) * 3600, 2)
    return f"{abs(d)}°{m}'{s}\""


def dms_to_decimal(degrees: float, minutes: float, seconds: float) -> float:
    """Convert DMS components to decimal degrees."""
    d, m, s = float(degrees), float(minutes), float(seconds)
    sign = -1 if d < 0 else 1
    return round(sign * (abs(d) + m / 60 + s / 3600), 6)


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing (forward azimuth) in degrees from point 1 to point 2."""
    la1, la2 = math.radians(float(lat1)), math.radians(float(lat2))
    dlon = math.radians(float(lon2) - float(lon1))
    x = math.sin(dlon) * math.cos(la2)
    y = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(dlon)
    brng = math.degrees(math.atan2(x, y))
    return round(brng % 360, 2)


def midpoint(lat1: float, lon1: float, lat2: float, lon2: float) -> tuple[float, float]:
    """Geographic midpoint between two lat/lon points."""
    la1, lo1 = math.radians(float(lat1)), math.radians(float(lon1))
    la2, lo2 = math.radians(float(lat2)), math.radians(float(lon2))
    dlon = lo2 - lo1
    bx = math.cos(la2) * math.cos(dlon)
    by = math.cos(la2) * math.sin(dlon)
    lat_mid = math.atan2(math.sin(la1) + math.sin(la2),
                         math.sqrt((math.cos(la1) + bx) ** 2 + by ** 2))
    lon_mid = lo1 + math.atan2(by, math.cos(la1) + bx)
    return (round(math.degrees(lat_mid), 6), round(math.degrees(lon_mid), 6))


def destination_point(lat: float, lon: float, bearing_deg: float, distance_km: float) -> tuple[float, float]:
    """Point at given distance and bearing from start point."""
    R = 6371.0
    la = math.radians(float(lat))
    lo = math.radians(float(lon))
    brng = math.radians(float(bearing_deg))
    d = float(distance_km) / R
    lat2 = math.asin(math.sin(la) * math.cos(d) + math.cos(la) * math.sin(d) * math.cos(brng))
    lon2 = lo + math.atan2(math.sin(brng) * math.sin(d) * math.cos(la),
                           math.cos(d) - math.sin(la) * math.sin(lat2))
    return (round(math.degrees(lat2), 6), round(math.degrees(lon2), 6))


def is_valid_coordinate(lat: float, lon: float) -> bool:
    """Check if lat/lon is within valid range (-90..90, -180..180)."""
    return -90 <= float(lat) <= 90 and -180 <= float(lon) <= 180


def compass_direction(bearing_deg: float) -> str:
    """Convert bearing in degrees to 16-point compass direction."""
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = round(float(bearing_deg) % 360 / 22.5) % 16
    return dirs[idx]


COORDINATE_FUNCTIONS = {
    "haversine": haversine,
    "haversine_miles": haversine_miles,
    "decimal_to_dms": decimal_to_dms,
    "dms_to_decimal": dms_to_decimal,
    "bearing": bearing,
    "midpoint": midpoint,
    "destination_point": destination_point,
    "is_valid_coordinate": is_valid_coordinate,
    "compass_direction": compass_direction,
}

COORDINATE_NL_PATTERNS = [
    (r'distance\s+(?:between|from)\s+\(?([-\d.]+)\s*,\s*([-\d.]+)\)?\s+(?:and|to)\s+\(?([-\d.]+)\s*,\s*([-\d.]+)\)?', 'haversine({0}, {1}, {2}, {3})'),
    (r'haversine.*?([-\d.]+)\s*,\s*([-\d.]+).*?([-\d.]+)\s*,\s*([-\d.]+)', 'haversine({0}, {1}, {2}, {3})'),
    (r'bearing\s+from\s+\(?([-\d.]+)\s*,\s*([-\d.]+)\)?\s+to\s+\(?([-\d.]+)\s*,\s*([-\d.]+)\)?', 'bearing({0}, {1}, {2}, {3})'),
    (r'convert\s+([-\d.]+)\s*(?:degrees?|°)\s+(\d+)\s*[\'′]\s*([\d.]+)\s*[\"″]?\s+to\s+decimal', 'dms_to_decimal({0}, {1}, {2})'),
    (r'convert\s+([-\d.]+)\s+(?:decimal\s+)?degrees?\s+to\s+dms', 'decimal_to_dms({0})'),
    (r'midpoint\s+(?:between|of)\s+\(?([-\d.]+)\s*,\s*([-\d.]+)\)?\s+(?:and)\s+\(?([-\d.]+)\s*,\s*([-\d.]+)\)?', 'midpoint({0}, {1}, {2}, {3})'),
]
