"""
CALM Advanced units backend — dimensional analysis, unit parsing, derived units.

Extends convert_ops with more conversions and unit math.
"""

from __future__ import annotations

import math


_CONVERSIONS = {
    # Length
    ("meter", "foot"): 3.28084, ("meter", "inch"): 39.3701,
    ("meter", "yard"): 1.09361, ("meter", "mile"): 0.000621371,
    ("meter", "kilometer"): 0.001, ("meter", "centimeter"): 100,
    ("meter", "millimeter"): 1000, ("meter", "micrometer"): 1e6,
    ("meter", "nanometer"): 1e9, ("meter", "nautical mile"): 0.000539957,
    ("meter", "light year"): 1.057e-16,
    # Mass
    ("kilogram", "pound"): 2.20462, ("kilogram", "ounce"): 35.274,
    ("kilogram", "gram"): 1000, ("kilogram", "milligram"): 1e6,
    ("kilogram", "ton"): 0.00110231, ("kilogram", "tonne"): 0.001,
    ("kilogram", "stone"): 0.157473,
    # Volume
    ("liter", "gallon"): 0.264172, ("liter", "quart"): 1.05669,
    ("liter", "pint"): 2.11338, ("liter", "cup"): 4.22675,
    ("liter", "fluid ounce"): 33.814, ("liter", "tablespoon"): 67.628,
    ("liter", "teaspoon"): 202.884, ("liter", "milliliter"): 1000,
    # Speed
    ("mps", "kmh"): 3.6, ("mps", "mph"): 2.23694,
    ("mps", "knot"): 1.94384, ("mps", "fps"): 3.28084,
    # Area
    ("sqm", "sqft"): 10.7639, ("sqm", "acre"): 0.000247105,
    ("sqm", "hectare"): 0.0001, ("sqm", "sqkm"): 1e-6,
    ("sqm", "sqmile"): 3.861e-7,
    # Pressure
    ("pascal", "bar"): 1e-5, ("pascal", "atm"): 9.8692e-6,
    ("pascal", "psi"): 0.000145038, ("pascal", "mmhg"): 0.00750062,
    ("pascal", "torr"): 0.00750062,
    # Energy
    ("joule", "calorie"): 0.239006, ("joule", "kilocalorie"): 0.000239006,
    ("joule", "kwh"): 2.778e-7, ("joule", "btu"): 0.000947817,
    ("joule", "ev"): 6.242e18,
    # Power
    ("watt", "horsepower"): 0.00134102, ("watt", "kilowatt"): 0.001,
    ("watt", "btu_per_hour"): 3.41214,
    # Temperature (special handling)
    # Data
    ("byte", "kilobyte"): 1/1024, ("byte", "megabyte"): 1/(1024**2),
    ("byte", "gigabyte"): 1/(1024**3), ("byte", "terabyte"): 1/(1024**4),
    ("byte", "bit"): 8, ("byte", "kibibyte"): 1/1024,
}

# Build reverse lookup
_ALL_CONVERSIONS = {}
for (a, b), factor in _CONVERSIONS.items():
    _ALL_CONVERSIONS[(a, b)] = factor
    _ALL_CONVERSIONS[(b, a)] = 1.0 / factor


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """Convert between units. E.g. convert(100, 'meter', 'foot')."""
    key = (str(from_unit).lower().strip(), str(to_unit).lower().strip())
    factor = _ALL_CONVERSIONS.get(key)
    if factor is None:
        return -1.0
    return round(float(value) * factor, 6)


def celsius_to_fahrenheit(c: float) -> float:
    return round(float(c) * 9/5 + 32, 4)


def fahrenheit_to_celsius(f: float) -> float:
    return round((float(f) - 32) * 5/9, 4)


def celsius_to_kelvin(c: float) -> float:
    return round(float(c) + 273.15, 4)


def kelvin_to_celsius(k: float) -> float:
    return round(float(k) - 273.15, 4)


def fahrenheit_to_kelvin(f: float) -> float:
    return round((float(f) - 32) * 5/9 + 273.15, 4)


def kelvin_to_fahrenheit(k: float) -> float:
    return round((float(k) - 273.15) * 9/5 + 32, 4)


def bmi(weight_kg: float, height_m: float) -> dict:
    """Body Mass Index: weight(kg) / height(m)²."""
    h = float(height_m)
    if h <= 0:
        return {"error": "invalid height"}
    val = round(float(weight_kg) / (h ** 2), 1)
    if val < 18.5: cat = "underweight"
    elif val < 25: cat = "normal"
    elif val < 30: cat = "overweight"
    else: cat = "obese"
    return {"bmi": val, "category": cat}


def fuel_consumption(liters: float, km: float) -> dict:
    """Fuel consumption in L/100km and MPG."""
    l, k = float(liters), float(km)
    if k <= 0 or l <= 0:
        return {"error": "invalid input"}
    l_per_100 = round(l / k * 100, 2)
    mpg = round(235.215 / l_per_100, 1)
    return {"l_per_100km": l_per_100, "mpg_us": mpg}


def speed_of_sound(temp_c: float = 20) -> float:
    """Speed of sound in air at given temperature (m/s). v ≈ 331.3 + 0.606T."""
    return round(331.3 + 0.606 * float(temp_c), 2)


def mach_to_speed(mach: float, temp_c: float = 20) -> float:
    """Convert Mach number to m/s."""
    return round(float(mach) * speed_of_sound(temp_c), 2)


def light_travel_time(distance_km: float) -> float:
    """Time for light to travel a distance in seconds. c = 299792.458 km/s."""
    return round(float(distance_km) / 299792.458, 6)


def dB_to_ratio(dB: float) -> float:
    """Convert decibels to power ratio."""
    return round(10 ** (float(dB) / 10), 6)


def ratio_to_dB(ratio: float) -> float:
    """Convert power ratio to decibels."""
    if float(ratio) <= 0:
        return float('-inf')
    return round(10 * math.log10(float(ratio)), 4)


def wavelength_to_frequency(wavelength_m: float) -> float:
    """Convert wavelength (meters) to frequency (Hz). f = c/λ."""
    w = float(wavelength_m)
    if w <= 0:
        return 0.0
    return round(299792458 / w, 2)


def frequency_to_wavelength(frequency_hz: float) -> float:
    """Convert frequency (Hz) to wavelength (meters). λ = c/f."""
    f = float(frequency_hz)
    if f <= 0:
        return 0.0
    return round(299792458 / f, 6)


UNITS_FUNCTIONS = {
    "convert": convert,
    "celsius_to_fahrenheit": celsius_to_fahrenheit,
    "fahrenheit_to_celsius": fahrenheit_to_celsius,
    "celsius_to_kelvin": celsius_to_kelvin,
    "kelvin_to_celsius": kelvin_to_celsius,
    "fahrenheit_to_kelvin": fahrenheit_to_kelvin,
    "kelvin_to_fahrenheit": kelvin_to_fahrenheit,
    "bmi": bmi,
    "fuel_consumption": fuel_consumption,
    "speed_of_sound": speed_of_sound,
    "mach_to_speed": mach_to_speed,
    "light_travel_time": light_travel_time,
    "dB_to_ratio": dB_to_ratio,
    "ratio_to_dB": ratio_to_dB,
    "wavelength_to_frequency": wavelength_to_frequency,
    "frequency_to_wavelength": frequency_to_wavelength,
}

UNITS_NL_PATTERNS = [
    (r'(?:convert|what is)\s+([\d.]+)\s+(meter|foot|inch|yard|mile|km|kilogram|pound|ounce|liter|gallon)s?\s+(?:to|in)\s+(\w+)', 'convert({0}, "{1}", "{2}")'),
    (r'(?:BMI|bmi)\s+(?:for|of|at)\s+([\d.]+)\s*(?:kg)?\s+(?:and|at|,)?\s*([\d.]+)\s*(?:m|meters?)?', 'bmi({0}, {1})'),
    (r'speed\s+of\s+sound\s+(?:at\s+)?([-\d.]+)\s*(?:°?C|celsius)?', 'speed_of_sound({0})'),
    (r'mach\s+([\d.]+)\s+(?:to|in)\s+(?:m/s|speed)', 'mach_to_speed({0})'),
    (r'([\d.]+)\s*(?:dB|decibels?)\s+(?:to|in|as)\s+ratio', 'dB_to_ratio({0})'),
    (r'wavelength\s+([\d.]+)\s*(?:m|nm|meters?)?\s+(?:to|in)\s+frequency', 'wavelength_to_frequency({0})'),
]
