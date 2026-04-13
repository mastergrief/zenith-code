"""
CALM Measurement knowledge backend — SI prefixes, unit relationships, physical units.

Models confuse milli/micro, mess up SI prefix powers, hallucinate unit conversions.
"""

from __future__ import annotations

_DATA_VERSION = "2025-01"

# (name, symbol, power_of_10)
_SI_PREFIXES = {
    "quetta": ("Q", 30),
    "ronna": ("R", 27),
    "yotta": ("Y", 24),
    "zetta": ("Z", 21),
    "exa": ("E", 18),
    "peta": ("P", 15),
    "tera": ("T", 12),
    "giga": ("G", 9),
    "mega": ("M", 6),
    "kilo": ("k", 3),
    "hecto": ("h", 2),
    "deca": ("da", 1),
    "deci": ("d", -1),
    "centi": ("c", -2),
    "milli": ("m", -3),
    "micro": ("μ", -6),
    "nano": ("n", -9),
    "pico": ("p", -12),
    "femto": ("f", -15),
    "atto": ("a", -18),
    "zepto": ("z", -21),
    "yocto": ("y", -24),
    "ronto": ("r", -27),
    "quecto": ("q", -30),
}

# (base_unit, dimension)
_SI_BASE_UNITS = {
    "meter": ("m", "length"),
    "kilogram": ("kg", "mass"),
    "second": ("s", "time"),
    "ampere": ("A", "electric current"),
    "kelvin": ("K", "temperature"),
    "mole": ("mol", "amount of substance"),
    "candela": ("cd", "luminous intensity"),
}

# (equivalent, base_units)
_DERIVED_UNITS = {
    "hertz": ("Hz", "s⁻¹", "frequency"),
    "newton": ("N", "kg·m·s⁻²", "force"),
    "pascal": ("Pa", "N·m⁻²", "pressure"),
    "joule": ("J", "N·m", "energy"),
    "watt": ("W", "J·s⁻¹", "power"),
    "coulomb": ("C", "A·s", "electric charge"),
    "volt": ("V", "W·A⁻¹", "voltage"),
    "farad": ("F", "C·V⁻¹", "capacitance"),
    "ohm": ("Ω", "V·A⁻¹", "resistance"),
    "siemens": ("S", "Ω⁻¹", "conductance"),
    "weber": ("Wb", "V·s", "magnetic flux"),
    "tesla": ("T", "Wb·m⁻²", "magnetic flux density"),
    "henry": ("H", "Wb·A⁻¹", "inductance"),
    "lux": ("lx", "lm·m⁻²", "illuminance"),
    "becquerel": ("Bq", "s⁻¹", "radioactivity"),
    "gray": ("Gy", "J·kg⁻¹", "absorbed dose"),
    "sievert": ("Sv", "J·kg⁻¹", "equivalent dose"),
    "katal": ("kat", "mol·s⁻¹", "catalytic activity"),
}

# Common non-SI conversions: (to_si_value, si_unit)
_COMMON_CONVERSIONS = {
    "inch": (0.0254, "meter"),
    "foot": (0.3048, "meter"),
    "yard": (0.9144, "meter"),
    "mile": (1609.344, "meter"),
    "nautical mile": (1852, "meter"),
    "acre": (4046.8564224, "square meter"),
    "hectare": (10000, "square meter"),
    "pound": (0.45359237, "kilogram"),
    "ounce": (0.028349523125, "kilogram"),
    "ton": (907.18474, "kilogram"),
    "tonne": (1000, "kilogram"),
    "gallon": (3.785411784, "liter"),
    "quart": (0.946352946, "liter"),
    "pint": (0.473176473, "liter"),
    "cup": (0.2365882365, "liter"),
    "tablespoon": (0.01478676478125, "liter"),
    "teaspoon": (0.00492892159375, "liter"),
    "fluid ounce": (0.0295735295625, "liter"),
    "bar": (100000, "pascal"),
    "atmosphere": (101325, "pascal"),
    "psi": (6894.757293168, "pascal"),
    "calorie": (4.184, "joule"),
    "kilocalorie": (4184, "joule"),
    "btu": (1055.06, "joule"),
    "horsepower": (745.7, "watt"),
    "knot": (0.514444, "meter per second"),
    "light year": (9.461e15, "meter"),
    "astronomical unit": (1.496e11, "meter"),
    "parsec": (3.086e16, "meter"),
}


def si_prefix(name: str) -> dict:
    """Get SI prefix details: symbol and power of 10."""
    entry = _SI_PREFIXES.get(str(name).lower())
    if not entry:
        return {"error": f"Unknown prefix: {name}"}
    return {"name": name, "symbol": entry[0], "power": entry[1], "factor": 10 ** entry[1]}


def si_base_unit(name: str) -> dict:
    """Get SI base unit details."""
    entry = _SI_BASE_UNITS.get(str(name).lower())
    if not entry:
        return {"error": f"Unknown base unit: {name}"}
    return {"name": name, "symbol": entry[0], "dimension": entry[1]}


def derived_unit(name: str) -> dict:
    """Get SI derived unit: symbol, base unit expression, dimension."""
    entry = _DERIVED_UNITS.get(str(name).lower())
    if not entry:
        return {"error": f"Unknown derived unit: {name}"}
    return {"name": name, "symbol": entry[0], "base_units": entry[1], "dimension": entry[2]}


def conversion_factor(unit: str) -> dict:
    """Get SI conversion factor for common non-SI units."""
    entry = _COMMON_CONVERSIONS.get(str(unit).lower())
    if not entry:
        return {"error": f"Unknown unit: {unit}"}
    return {"unit": unit, "si_value": entry[0], "si_unit": entry[1]}


def list_prefixes() -> list[dict]:
    """List all SI prefixes ordered by power."""
    return sorted(
        [{"name": n, "symbol": v[0], "power": v[1]} for n, v in _SI_PREFIXES.items()],
        key=lambda x: x["power"], reverse=True
    )


def prefix_between(from_prefix: str, to_prefix: str) -> float:
    """Conversion factor between two SI prefixes (e.g. kilo→milli = 1e6)."""
    f = _SI_PREFIXES.get(str(from_prefix).lower())
    t = _SI_PREFIXES.get(str(to_prefix).lower())
    if not f or not t:
        return -1.0
    return 10.0 ** (f[1] - t[1])


MEASUREMENT_FUNCTIONS = {
    "si_prefix": si_prefix,
    "si_base_unit": si_base_unit,
    "derived_unit": derived_unit,
    "conversion_factor": conversion_factor,
    "list_prefixes": list_prefixes,
    "prefix_between": prefix_between,
}

MEASUREMENT_NL_PATTERNS = [
    (r'(?:what is|what\'s) (?:a |the )?(\w+) (?:prefix|si prefix)', 'si_prefix("{0}")'),
    (r'(?:what is|what\'s) (?:a |the )?(\w+) in si', 'conversion_factor("{0}")'),
    (r'(?:how many|convert) (\w+) (?:in|to|per) (\w+).*?(?:prefix|si)', 'prefix_between("{0}", "{1}")'),
]
