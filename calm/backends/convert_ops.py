"""
CALM unit conversion backend — verified measurement conversions.

The model writes "5 miles is 8.05 km" and Auto-CALM verifies on CPU.

Domains: length, weight, temperature, volume, speed, data, time.
"""

from __future__ import annotations

from typing import Union

# Conversion tables — all relative to a base unit per domain.
# Length: meters
_LENGTH = {
    "m": 1.0, "meter": 1.0, "meters": 1.0,
    "km": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0,
    "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01,
    "mm": 0.001, "millimeter": 0.001, "millimeters": 0.001,
    "mi": 1609.344, "mile": 1609.344, "miles": 1609.344,
    "yd": 0.9144, "yard": 0.9144, "yards": 0.9144,
    "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
    "in": 0.0254, "inch": 0.0254, "inches": 0.0254,
    "nm": 1852.0, "nautical_mile": 1852.0,
}

# Weight: grams
_WEIGHT = {
    "g": 1.0, "gram": 1.0, "grams": 1.0,
    "kg": 1000.0, "kilogram": 1000.0, "kilograms": 1000.0,
    "mg": 0.001, "milligram": 0.001, "milligrams": 0.001,
    "lb": 453.592, "pound": 453.592, "pounds": 453.592,
    "oz": 28.3495, "ounce": 28.3495, "ounces": 28.3495,
    "ton": 907185.0, "tons": 907185.0,
    "tonne": 1000000.0, "tonnes": 1000000.0,
}

# Volume: liters
_VOLUME = {
    "l": 1.0, "liter": 1.0, "liters": 1.0, "litre": 1.0,
    "ml": 0.001, "milliliter": 0.001, "milliliters": 0.001,
    "gal": 3.78541, "gallon": 3.78541, "gallons": 3.78541,
    "qt": 0.946353, "quart": 0.946353, "quarts": 0.946353,
    "pt": 0.473176, "pint": 0.473176, "pints": 0.473176,
    "cup": 0.236588, "cups": 0.236588,
    "fl_oz": 0.0295735, "fluid_ounce": 0.0295735,
    "tbsp": 0.0147868, "tablespoon": 0.0147868,
    "tsp": 0.00492892, "teaspoon": 0.00492892,
}

# Speed: m/s
_SPEED = {
    "m/s": 1.0, "mps": 1.0,
    "km/h": 1 / 3.6, "kmh": 1 / 3.6, "kph": 1 / 3.6,
    "mph": 0.44704,
    "knot": 0.514444, "knots": 0.514444,
    "ft/s": 0.3048, "fps": 0.3048,
}

# Data: bytes
_DATA = {
    "b": 1, "byte": 1, "bytes": 1,
    "kb": 1024, "kilobyte": 1024, "kilobytes": 1024,
    "mb": 1024**2, "megabyte": 1024**2, "megabytes": 1024**2,
    "gb": 1024**3, "gigabyte": 1024**3, "gigabytes": 1024**3,
    "tb": 1024**4, "terabyte": 1024**4, "terabytes": 1024**4,
    "pb": 1024**5, "petabyte": 1024**5,
    "bit": 0.125, "bits": 0.125,
    "kbit": 128, "mbit": 128 * 1024, "gbit": 128 * 1024**2,
}

# Time: seconds
_TIME = {
    "s": 1, "sec": 1, "second": 1, "seconds": 1,
    "ms": 0.001, "millisecond": 0.001, "milliseconds": 0.001,
    "us": 1e-6, "microsecond": 1e-6, "microseconds": 1e-6,
    "ns": 1e-9, "nanosecond": 1e-9,
    "min": 60, "minute": 60, "minutes": 60,
    "h": 3600, "hr": 3600, "hour": 3600, "hours": 3600,
    "day": 86400, "days": 86400,
    "week": 604800, "weeks": 604800,
    "year": 31557600, "years": 31557600,  # Julian year
}

_DOMAINS = {
    "length": _LENGTH, "weight": _WEIGHT, "volume": _VOLUME,
    "speed": _SPEED, "data": _DATA, "time": _TIME,
}


def _find_domain(unit: str) -> tuple:
    """Find which domain a unit belongs to. Returns (domain_name, table)."""
    unit_lower = unit.lower().strip()
    for name, table in _DOMAINS.items():
        if unit_lower in table:
            return name, table
    raise ValueError(f"unknown unit: {unit}")


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a value between units. Auto-detects domain."""
    from_domain, from_table = _find_domain(from_unit)
    to_domain, to_table = _find_domain(to_unit)
    if from_domain != to_domain:
        raise ValueError(f"can't convert {from_domain} to {to_domain}")

    # Convert to base unit, then to target.
    base = float(value) * from_table[from_unit.lower().strip()]
    return base / to_table[to_unit.lower().strip()]


def celsius_to_fahrenheit(c: float) -> float:
    return float(c) * 9 / 5 + 32


def fahrenheit_to_celsius(f: float) -> float:
    return (float(f) - 32) * 5 / 9


def celsius_to_kelvin(c: float) -> float:
    return float(c) + 273.15


def kelvin_to_celsius(k: float) -> float:
    return float(k) - 273.15


CONVERT_FUNCTIONS = {
    "convert": convert,
    "celsius_to_fahrenheit": celsius_to_fahrenheit,
    "fahrenheit_to_celsius": fahrenheit_to_celsius,
    "celsius_to_kelvin": celsius_to_kelvin,
    "kelvin_to_celsius": kelvin_to_celsius,
}
