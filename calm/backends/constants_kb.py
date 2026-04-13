"""
CALM Physical Constants knowledge backend.

Models get the exponent wrong, mix up units, confuse similar constants.
CODATA 2018 recommended values.
"""

from __future__ import annotations

# (value, unit, description)
_CONSTANTS = {
    "speed_of_light": (299_792_458, "m/s", "Speed of light in vacuum"),
    "c": (299_792_458, "m/s", "Speed of light in vacuum"),
    "planck": (6.62607015e-34, "J·s", "Planck constant"),
    "h": (6.62607015e-34, "J·s", "Planck constant"),
    "hbar": (1.054571817e-34, "J·s", "Reduced Planck constant"),
    "boltzmann": (1.380649e-23, "J/K", "Boltzmann constant"),
    "k_b": (1.380649e-23, "J/K", "Boltzmann constant"),
    "avogadro": (6.02214076e23, "mol⁻¹", "Avogadro constant"),
    "n_a": (6.02214076e23, "mol⁻¹", "Avogadro constant"),
    "gravitational": (6.67430e-11, "m³/(kg·s²)", "Gravitational constant"),
    "g_const": (6.67430e-11, "m³/(kg·s²)", "Gravitational constant"),
    "gravity": (9.80665, "m/s²", "Standard gravity"),
    "g": (9.80665, "m/s²", "Standard gravity"),
    "electron_mass": (9.1093837015e-31, "kg", "Electron mass"),
    "proton_mass": (1.67262192369e-27, "kg", "Proton mass"),
    "neutron_mass": (1.67492749804e-27, "kg", "Neutron mass"),
    "elementary_charge": (1.602176634e-19, "C", "Elementary charge"),
    "e_charge": (1.602176634e-19, "C", "Elementary charge"),
    "vacuum_permittivity": (8.8541878128e-12, "F/m", "Vacuum permittivity (ε₀)"),
    "epsilon_0": (8.8541878128e-12, "F/m", "Vacuum permittivity (ε₀)"),
    "vacuum_permeability": (1.25663706212e-6, "N/A²", "Vacuum permeability (μ₀)"),
    "mu_0": (1.25663706212e-6, "N/A²", "Vacuum permeability (μ₀)"),
    "gas_constant": (8.314462618, "J/(mol·K)", "Ideal gas constant"),
    "r_gas": (8.314462618, "J/(mol·K)", "Ideal gas constant"),
    "stefan_boltzmann": (5.670374419e-8, "W/(m²·K⁴)", "Stefan-Boltzmann constant"),
    "fine_structure": (7.2973525693e-3, "", "Fine-structure constant (α)"),
    "alpha": (7.2973525693e-3, "", "Fine-structure constant (α)"),
    "bohr_radius": (5.29177210903e-11, "m", "Bohr radius"),
    "rydberg": (10973731.568160, "m⁻¹", "Rydberg constant"),
    "electron_volt": (1.602176634e-19, "J", "1 eV in joules"),
    "ev": (1.602176634e-19, "J", "1 eV in joules"),
    "atomic_mass_unit": (1.66053906660e-27, "kg", "Atomic mass unit (u)"),
    "amu": (1.66053906660e-27, "kg", "Atomic mass unit (u)"),
    "faraday": (96485.33212, "C/mol", "Faraday constant"),
    "light_year": (9.4607e15, "m", "1 light-year in meters"),
    "au": (1.495978707e11, "m", "1 astronomical unit in meters"),
    "parsec": (3.0857e16, "m", "1 parsec in meters"),
    "absolute_zero": (-273.15, "°C", "Absolute zero"),
    "water_freezing": (273.15, "K", "Water freezing point"),
    "water_boiling": (373.15, "K", "Water boiling point (at 1 atm)"),
    "atmosphere": (101325, "Pa", "Standard atmosphere"),
    "atm": (101325, "Pa", "Standard atmosphere"),
}

# Aliases for natural language
_ALIASES = {
    "speed of light": "speed_of_light",
    "planck constant": "planck",
    "planck's constant": "planck",
    "boltzmann constant": "boltzmann",
    "boltzmann's constant": "boltzmann",
    "avogadro's number": "avogadro",
    "avogadro number": "avogadro",
    "gravitational constant": "gravitational",
    "electron mass": "electron_mass",
    "proton mass": "proton_mass",
    "neutron mass": "neutron_mass",
    "elementary charge": "elementary_charge",
    "gas constant": "gas_constant",
    "fine structure constant": "fine_structure",
    "bohr radius": "bohr_radius",
    "rydberg constant": "rydberg",
    "faraday constant": "faraday",
    "stefan boltzmann constant": "stefan_boltzmann",
    "stefan-boltzmann constant": "stefan_boltzmann",
    "absolute zero": "absolute_zero",
    "standard gravity": "gravity",
    "standard atmosphere": "atmosphere",
}


def _resolve(name: str) -> str | None:
    key = name.strip().lower().replace("-", "_").replace("'", "")
    key = _ALIASES.get(key, key)
    return key if key in _CONSTANTS else None


def physical_constant(name: str) -> str:
    """Look up a physical constant by name. Returns value with unit."""
    key = _resolve(name)
    if not key:
        return f"unknown constant: {name}"
    val, unit, desc = _CONSTANTS[key]
    return f"{val} {unit}" if unit else str(val)


def constant_value(name: str) -> float:
    """Numeric value of a physical constant."""
    key = _resolve(name)
    if not key:
        return -1.0
    return _CONSTANTS[key][0]


def constant_unit(name: str) -> str:
    """Unit of a physical constant."""
    key = _resolve(name)
    if not key:
        return f"unknown constant: {name}"
    return _CONSTANTS[key][1]


def constant_info(name: str) -> str:
    """Full info: value, unit, and description."""
    key = _resolve(name)
    if not key:
        return f"unknown constant: {name}"
    val, unit, desc = _CONSTANTS[key]
    return f"{desc}: {val} {unit}"


def list_constants() -> list:
    """List all available constant names (deduplicated)."""
    seen = set()
    result = []
    for key in _CONSTANTS:
        desc = _CONSTANTS[key][2]
        if desc not in seen:
            seen.add(desc)
            result.append(f"{key}: {desc}")
    return result


CONSTANTS_NL_PATTERNS = [
    (r'(speed of light|planck.s? constant|boltzmann.s? constant|avogadro.s? number|gravitational constant|elementary charge|gas constant|fine.structure constant|bohr radius)', 'physical_constant("{0}")'),
]

CONSTANTS_FUNCTIONS = {
    "physical_constant": physical_constant,
    "constant_value": constant_value,
    "constant_unit": constant_unit,
    "constant_info": constant_info,
    "list_constants": list_constants,
}
