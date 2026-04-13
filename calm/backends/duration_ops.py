"""
CALM Duration backend — parse, format, convert time durations.

Models mix up units and can't do duration arithmetic. Pure re+math.
"""

from __future__ import annotations

import re


def duration_parse(s: str) -> int:
    """Parse duration string to total seconds. Supports: '2h30m', '1d12h', '90s', '1h 30m 45s', ISO 8601 'PT1H30M'."""
    s = str(s).strip()
    total = 0

    # ISO 8601: PT1H30M45S
    iso = re.match(r'^P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?$', s, re.IGNORECASE)
    if iso and any(iso.groups()):
        if iso.group(1):
            total += int(iso.group(1)) * 86400
        if iso.group(2):
            total += int(iso.group(2)) * 3600
        if iso.group(3):
            total += int(iso.group(3)) * 60
        if iso.group(4):
            total += int(float(iso.group(4)))
        return total

    # Human-readable: 2h30m, 1d 12h, 90s, 3h 45m 12s
    units = {"d": 86400, "h": 3600, "m": 60, "s": 1}
    matches = re.findall(r'(\d+(?:\.\d+)?)\s*([dhms])', s, re.IGNORECASE)
    if matches:
        for val, unit in matches:
            total += int(float(val)) * units.get(unit.lower(), 0)
        return total

    # Plain number = seconds
    try:
        return int(float(s))
    except ValueError:
        return -1


def duration_format(seconds: int) -> str:
    """Format seconds as human-readable duration (e.g. '2h 30m 45s')."""
    seconds = int(seconds)
    if seconds < 0:
        return f"-{duration_format(-seconds)}"
    if seconds == 0:
        return "0s"
    parts = []
    for unit, size in [("d", 86400), ("h", 3600), ("m", 60), ("s", 1)]:
        if seconds >= size:
            count = seconds // size
            seconds %= size
            parts.append(f"{count}{unit}")
    return " ".join(parts)


def duration_to_iso(seconds: int) -> str:
    """Convert seconds to ISO 8601 duration (e.g. 'PT2H30M45S')."""
    seconds = int(seconds)
    if seconds == 0:
        return "PT0S"
    parts = ["P"]
    days = seconds // 86400
    seconds %= 86400
    if days:
        parts.append(f"{days}D")
    if seconds > 0:
        parts.append("T")
        hours = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        secs = seconds % 60
        if hours:
            parts.append(f"{hours}H")
        if minutes:
            parts.append(f"{minutes}M")
        if secs:
            parts.append(f"{secs}S")
    return "".join(parts)


def duration_convert(value: int, from_unit: str, to_unit: str) -> str:
    """Convert between time units (seconds, minutes, hours, days, weeks)."""
    units = {
        "s": 1, "sec": 1, "second": 1, "seconds": 1,
        "m": 60, "min": 60, "minute": 60, "minutes": 60,
        "h": 3600, "hr": 3600, "hour": 3600, "hours": 3600,
        "d": 86400, "day": 86400, "days": 86400,
        "w": 604800, "week": 604800, "weeks": 604800,
    }
    f = units.get(str(from_unit).lower())
    t = units.get(str(to_unit).lower())
    if not f:
        return f"unknown unit: {from_unit}"
    if not t:
        return f"unknown unit: {to_unit}"
    total_seconds = int(value) * f
    result = total_seconds / t
    if result == int(result):
        return f"{int(result)} {to_unit}"
    return f"{result:.4f} {to_unit}"


def duration_add(d1: str, d2: str) -> str:
    """Add two duration strings. Returns human-readable result."""
    s1 = duration_parse(d1)
    s2 = duration_parse(d2)
    if s1 < 0 or s2 < 0:
        return "parse error"
    return duration_format(s1 + s2)


def duration_subtract(d1: str, d2: str) -> str:
    """Subtract d2 from d1. Returns human-readable result."""
    s1 = duration_parse(d1)
    s2 = duration_parse(d2)
    if s1 < 0 or s2 < 0:
        return "parse error"
    return duration_format(s1 - s2)


def seconds_in(value: int, unit: str) -> int:
    """How many seconds in N units (e.g. seconds_in(3, 'hours') → 10800)."""
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800,
             "second": 1, "seconds": 1, "minute": 60, "minutes": 60,
             "hour": 3600, "hours": 3600, "day": 86400, "days": 86400,
             "week": 604800, "weeks": 604800}
    mult = units.get(str(unit).lower(), 0)
    return int(value) * mult


DURATION_FUNCTIONS = {
    "duration_parse": duration_parse,
    "duration_format": duration_format,
    "duration_to_iso": duration_to_iso,
    "duration_convert": duration_convert,
    "duration_add": duration_add,
    "duration_subtract": duration_subtract,
    "seconds_in": seconds_in,
}
