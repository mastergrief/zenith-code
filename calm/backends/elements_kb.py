"""
CALM Periodic Table knowledge backend — elements, symbols, weights.

Models transpose atomic numbers, confuse similar elements, get weights wrong.
All 118 elements. Data is stable (last element added 2016).
"""

from __future__ import annotations

# (symbol, name, atomic_weight, group, period, category, electron_config_short)
_ELEMENTS = {
    1: ("H", "Hydrogen", 1.008, 1, 1, "Nonmetal", "1s1"),
    2: ("He", "Helium", 4.003, 18, 1, "Noble Gas", "1s2"),
    3: ("Li", "Lithium", 6.941, 1, 2, "Alkali Metal", "[He] 2s1"),
    4: ("Be", "Beryllium", 9.012, 2, 2, "Alkaline Earth", "[He] 2s2"),
    5: ("B", "Boron", 10.81, 13, 2, "Metalloid", "[He] 2s2 2p1"),
    6: ("C", "Carbon", 12.011, 14, 2, "Nonmetal", "[He] 2s2 2p2"),
    7: ("N", "Nitrogen", 14.007, 15, 2, "Nonmetal", "[He] 2s2 2p3"),
    8: ("O", "Oxygen", 15.999, 16, 2, "Nonmetal", "[He] 2s2 2p4"),
    9: ("F", "Fluorine", 18.998, 17, 2, "Halogen", "[He] 2s2 2p5"),
    10: ("Ne", "Neon", 20.180, 18, 2, "Noble Gas", "[He] 2s2 2p6"),
    11: ("Na", "Sodium", 22.990, 1, 3, "Alkali Metal", "[Ne] 3s1"),
    12: ("Mg", "Magnesium", 24.305, 2, 3, "Alkaline Earth", "[Ne] 3s2"),
    13: ("Al", "Aluminum", 26.982, 13, 3, "Post-Transition Metal", "[Ne] 3s2 3p1"),
    14: ("Si", "Silicon", 28.086, 14, 3, "Metalloid", "[Ne] 3s2 3p2"),
    15: ("P", "Phosphorus", 30.974, 15, 3, "Nonmetal", "[Ne] 3s2 3p3"),
    16: ("S", "Sulfur", 32.065, 16, 3, "Nonmetal", "[Ne] 3s2 3p4"),
    17: ("Cl", "Chlorine", 35.453, 17, 3, "Halogen", "[Ne] 3s2 3p5"),
    18: ("Ar", "Argon", 39.948, 18, 3, "Noble Gas", "[Ne] 3s2 3p6"),
    19: ("K", "Potassium", 39.098, 1, 4, "Alkali Metal", "[Ar] 4s1"),
    20: ("Ca", "Calcium", 40.078, 2, 4, "Alkaline Earth", "[Ar] 4s2"),
    21: ("Sc", "Scandium", 44.956, 3, 4, "Transition Metal", "[Ar] 3d1 4s2"),
    22: ("Ti", "Titanium", 47.867, 4, 4, "Transition Metal", "[Ar] 3d2 4s2"),
    23: ("V", "Vanadium", 50.942, 5, 4, "Transition Metal", "[Ar] 3d3 4s2"),
    24: ("Cr", "Chromium", 51.996, 6, 4, "Transition Metal", "[Ar] 3d5 4s1"),
    25: ("Mn", "Manganese", 54.938, 7, 4, "Transition Metal", "[Ar] 3d5 4s2"),
    26: ("Fe", "Iron", 55.845, 8, 4, "Transition Metal", "[Ar] 3d6 4s2"),
    27: ("Co", "Cobalt", 58.933, 9, 4, "Transition Metal", "[Ar] 3d7 4s2"),
    28: ("Ni", "Nickel", 58.693, 10, 4, "Transition Metal", "[Ar] 3d8 4s2"),
    29: ("Cu", "Copper", 63.546, 11, 4, "Transition Metal", "[Ar] 3d10 4s1"),
    30: ("Zn", "Zinc", 65.38, 12, 4, "Transition Metal", "[Ar] 3d10 4s2"),
    31: ("Ga", "Gallium", 69.723, 13, 4, "Post-Transition Metal", "[Ar] 3d10 4s2 4p1"),
    32: ("Ge", "Germanium", 72.630, 14, 4, "Metalloid", "[Ar] 3d10 4s2 4p2"),
    33: ("As", "Arsenic", 74.922, 15, 4, "Metalloid", "[Ar] 3d10 4s2 4p3"),
    34: ("Se", "Selenium", 78.971, 16, 4, "Nonmetal", "[Ar] 3d10 4s2 4p4"),
    35: ("Br", "Bromine", 79.904, 17, 4, "Halogen", "[Ar] 3d10 4s2 4p5"),
    36: ("Kr", "Krypton", 83.798, 18, 4, "Noble Gas", "[Ar] 3d10 4s2 4p6"),
    37: ("Rb", "Rubidium", 85.468, 1, 5, "Alkali Metal", "[Kr] 5s1"),
    38: ("Sr", "Strontium", 87.62, 2, 5, "Alkaline Earth", "[Kr] 5s2"),
    39: ("Y", "Yttrium", 88.906, 3, 5, "Transition Metal", "[Kr] 4d1 5s2"),
    40: ("Zr", "Zirconium", 91.224, 4, 5, "Transition Metal", "[Kr] 4d2 5s2"),
    41: ("Nb", "Niobium", 92.906, 5, 5, "Transition Metal", "[Kr] 4d4 5s1"),
    42: ("Mo", "Molybdenum", 95.95, 6, 5, "Transition Metal", "[Kr] 4d5 5s1"),
    43: ("Tc", "Technetium", 98.0, 7, 5, "Transition Metal", "[Kr] 4d5 5s2"),
    44: ("Ru", "Ruthenium", 101.07, 8, 5, "Transition Metal", "[Kr] 4d7 5s1"),
    45: ("Rh", "Rhodium", 102.91, 9, 5, "Transition Metal", "[Kr] 4d8 5s1"),
    46: ("Pd", "Palladium", 106.42, 10, 5, "Transition Metal", "[Kr] 4d10"),
    47: ("Ag", "Silver", 107.87, 11, 5, "Transition Metal", "[Kr] 4d10 5s1"),
    48: ("Cd", "Cadmium", 112.41, 12, 5, "Transition Metal", "[Kr] 4d10 5s2"),
    49: ("In", "Indium", 114.82, 13, 5, "Post-Transition Metal", "[Kr] 4d10 5s2 5p1"),
    50: ("Sn", "Tin", 118.71, 14, 5, "Post-Transition Metal", "[Kr] 4d10 5s2 5p2"),
    51: ("Sb", "Antimony", 121.76, 15, 5, "Metalloid", "[Kr] 4d10 5s2 5p3"),
    52: ("Te", "Tellurium", 127.60, 16, 5, "Metalloid", "[Kr] 4d10 5s2 5p4"),
    53: ("I", "Iodine", 126.90, 17, 5, "Halogen", "[Kr] 4d10 5s2 5p5"),
    54: ("Xe", "Xenon", 131.29, 18, 5, "Noble Gas", "[Kr] 4d10 5s2 5p6"),
    55: ("Cs", "Cesium", 132.91, 1, 6, "Alkali Metal", "[Xe] 6s1"),
    56: ("Ba", "Barium", 137.33, 2, 6, "Alkaline Earth", "[Xe] 6s2"),
    57: ("La", "Lanthanum", 138.91, 3, 6, "Lanthanide", "[Xe] 5d1 6s2"),
    58: ("Ce", "Cerium", 140.12, 3, 6, "Lanthanide", "[Xe] 4f1 5d1 6s2"),
    59: ("Pr", "Praseodymium", 140.91, 3, 6, "Lanthanide", "[Xe] 4f3 6s2"),
    60: ("Nd", "Neodymium", 144.24, 3, 6, "Lanthanide", "[Xe] 4f4 6s2"),
    61: ("Pm", "Promethium", 145.0, 3, 6, "Lanthanide", "[Xe] 4f5 6s2"),
    62: ("Sm", "Samarium", 150.36, 3, 6, "Lanthanide", "[Xe] 4f6 6s2"),
    63: ("Eu", "Europium", 151.96, 3, 6, "Lanthanide", "[Xe] 4f7 6s2"),
    64: ("Gd", "Gadolinium", 157.25, 3, 6, "Lanthanide", "[Xe] 4f7 5d1 6s2"),
    65: ("Tb", "Terbium", 158.93, 3, 6, "Lanthanide", "[Xe] 4f9 6s2"),
    66: ("Dy", "Dysprosium", 162.50, 3, 6, "Lanthanide", "[Xe] 4f10 6s2"),
    67: ("Ho", "Holmium", 164.93, 3, 6, "Lanthanide", "[Xe] 4f11 6s2"),
    68: ("Er", "Erbium", 167.26, 3, 6, "Lanthanide", "[Xe] 4f12 6s2"),
    69: ("Tm", "Thulium", 168.93, 3, 6, "Lanthanide", "[Xe] 4f13 6s2"),
    70: ("Yb", "Ytterbium", 173.05, 3, 6, "Lanthanide", "[Xe] 4f14 6s2"),
    71: ("Lu", "Lutetium", 174.97, 3, 6, "Lanthanide", "[Xe] 4f14 5d1 6s2"),
    72: ("Hf", "Hafnium", 178.49, 4, 6, "Transition Metal", "[Xe] 4f14 5d2 6s2"),
    73: ("Ta", "Tantalum", 180.95, 5, 6, "Transition Metal", "[Xe] 4f14 5d3 6s2"),
    74: ("W", "Tungsten", 183.84, 6, 6, "Transition Metal", "[Xe] 4f14 5d4 6s2"),
    75: ("Re", "Rhenium", 186.21, 7, 6, "Transition Metal", "[Xe] 4f14 5d5 6s2"),
    76: ("Os", "Osmium", 190.23, 8, 6, "Transition Metal", "[Xe] 4f14 5d6 6s2"),
    77: ("Ir", "Iridium", 192.22, 9, 6, "Transition Metal", "[Xe] 4f14 5d7 6s2"),
    78: ("Pt", "Platinum", 195.08, 10, 6, "Transition Metal", "[Xe] 4f14 5d9 6s1"),
    79: ("Au", "Gold", 199.97, 11, 6, "Transition Metal", "[Xe] 4f14 5d10 6s1"),
    80: ("Hg", "Mercury", 200.59, 12, 6, "Transition Metal", "[Xe] 4f14 5d10 6s2"),
    81: ("Tl", "Thallium", 204.38, 13, 6, "Post-Transition Metal", "[Xe] 4f14 5d10 6s2 6p1"),
    82: ("Pb", "Lead", 207.2, 14, 6, "Post-Transition Metal", "[Xe] 4f14 5d10 6s2 6p2"),
    83: ("Bi", "Bismuth", 208.98, 15, 6, "Post-Transition Metal", "[Xe] 4f14 5d10 6s2 6p3"),
    84: ("Po", "Polonium", 209.0, 16, 6, "Post-Transition Metal", "[Xe] 4f14 5d10 6s2 6p4"),
    85: ("At", "Astatine", 210.0, 17, 6, "Halogen", "[Xe] 4f14 5d10 6s2 6p5"),
    86: ("Rn", "Radon", 222.0, 18, 6, "Noble Gas", "[Xe] 4f14 5d10 6s2 6p6"),
    87: ("Fr", "Francium", 223.0, 1, 7, "Alkali Metal", "[Rn] 7s1"),
    88: ("Ra", "Radium", 226.0, 2, 7, "Alkaline Earth", "[Rn] 7s2"),
    89: ("Ac", "Actinium", 227.0, 3, 7, "Actinide", "[Rn] 6d1 7s2"),
    90: ("Th", "Thorium", 232.04, 3, 7, "Actinide", "[Rn] 6d2 7s2"),
    91: ("Pa", "Protactinium", 231.04, 3, 7, "Actinide", "[Rn] 5f2 6d1 7s2"),
    92: ("U", "Uranium", 238.03, 3, 7, "Actinide", "[Rn] 5f3 6d1 7s2"),
    93: ("Np", "Neptunium", 237.0, 3, 7, "Actinide", "[Rn] 5f4 6d1 7s2"),
    94: ("Pu", "Plutonium", 244.0, 3, 7, "Actinide", "[Rn] 5f6 7s2"),
    95: ("Am", "Americium", 243.0, 3, 7, "Actinide", "[Rn] 5f7 7s2"),
    96: ("Cm", "Curium", 247.0, 3, 7, "Actinide", "[Rn] 5f7 6d1 7s2"),
    97: ("Bk", "Berkelium", 247.0, 3, 7, "Actinide", "[Rn] 5f9 7s2"),
    98: ("Cf", "Californium", 251.0, 3, 7, "Actinide", "[Rn] 5f10 7s2"),
    99: ("Es", "Einsteinium", 252.0, 3, 7, "Actinide", "[Rn] 5f11 7s2"),
    100: ("Fm", "Fermium", 257.0, 3, 7, "Actinide", "[Rn] 5f12 7s2"),
    101: ("Md", "Mendelevium", 258.0, 3, 7, "Actinide", "[Rn] 5f13 7s2"),
    102: ("No", "Nobelium", 259.0, 3, 7, "Actinide", "[Rn] 5f14 7s2"),
    103: ("Lr", "Lawrencium", 266.0, 3, 7, "Actinide", "[Rn] 5f14 7s2 7p1"),
    104: ("Rf", "Rutherfordium", 267.0, 4, 7, "Transition Metal", "[Rn] 5f14 6d2 7s2"),
    105: ("Db", "Dubnium", 268.0, 5, 7, "Transition Metal", "[Rn] 5f14 6d3 7s2"),
    106: ("Sg", "Seaborgium", 269.0, 6, 7, "Transition Metal", "[Rn] 5f14 6d4 7s2"),
    107: ("Bh", "Bohrium", 270.0, 7, 7, "Transition Metal", "[Rn] 5f14 6d5 7s2"),
    108: ("Hs", "Hassium", 277.0, 8, 7, "Transition Metal", "[Rn] 5f14 6d6 7s2"),
    109: ("Mt", "Meitnerium", 278.0, 9, 7, "Transition Metal", "[Rn] 5f14 6d7 7s2"),
    110: ("Ds", "Darmstadtium", 281.0, 10, 7, "Transition Metal", "[Rn] 5f14 6d8 7s2"),
    111: ("Rg", "Roentgenium", 282.0, 11, 7, "Transition Metal", "[Rn] 5f14 6d9 7s2"),
    112: ("Cn", "Copernicium", 285.0, 12, 7, "Transition Metal", "[Rn] 5f14 6d10 7s2"),
    113: ("Nh", "Nihonium", 286.0, 13, 7, "Post-Transition Metal", "[Rn] 5f14 6d10 7s2 7p1"),
    114: ("Fl", "Flerovium", 289.0, 14, 7, "Post-Transition Metal", "[Rn] 5f14 6d10 7s2 7p2"),
    115: ("Mc", "Moscovium", 290.0, 15, 7, "Post-Transition Metal", "[Rn] 5f14 6d10 7s2 7p3"),
    116: ("Lv", "Livermorium", 293.0, 16, 7, "Post-Transition Metal", "[Rn] 5f14 6d10 7s2 7p4"),
    117: ("Ts", "Tennessine", 294.0, 17, 7, "Halogen", "[Rn] 5f14 6d10 7s2 7p5"),
    118: ("Og", "Oganesson", 294.0, 18, 7, "Noble Gas", "[Rn] 5f14 6d10 7s2 7p6"),
}

# Reverse lookups: symbol → number, name → number
_BY_SYMBOL = {v[0].lower(): k for k, v in _ELEMENTS.items()}
_BY_NAME = {v[1].lower(): k for k, v in _ELEMENTS.items()}


def _resolve(element: str) -> int | None:
    """Resolve element by number, symbol, or name."""
    s = str(element).strip()
    try:
        n = int(s)
        return n if n in _ELEMENTS else None
    except ValueError:
        pass
    low = s.lower()
    return _BY_SYMBOL.get(low) or _BY_NAME.get(low)


def element_symbol(element: str) -> str:
    """Chemical symbol for an element (by number, symbol, or name)."""
    n = _resolve(element)
    return _ELEMENTS[n][0] if n else f"unknown element: {element}"


def element_name(element: str) -> str:
    """Full name of an element."""
    n = _resolve(element)
    return _ELEMENTS[n][1] if n else f"unknown element: {element}"


def atomic_number(element: str) -> int:
    """Atomic number of an element."""
    n = _resolve(element)
    return n if n else -1


def atomic_weight(element: str) -> float:
    """Standard atomic weight."""
    n = _resolve(element)
    return _ELEMENTS[n][2] if n else -1.0


def element_group(element: str) -> int:
    """Periodic table group (1-18)."""
    n = _resolve(element)
    return _ELEMENTS[n][3] if n else -1


def element_period(element: str) -> int:
    """Periodic table period (1-7)."""
    n = _resolve(element)
    return _ELEMENTS[n][4] if n else -1


def element_category(element: str) -> str:
    """Element category (Nonmetal, Noble Gas, Transition Metal, etc.)."""
    n = _resolve(element)
    return _ELEMENTS[n][5] if n else f"unknown element: {element}"


def electron_config(element: str) -> str:
    """Abbreviated electron configuration."""
    n = _resolve(element)
    return _ELEMENTS[n][6] if n else f"unknown element: {element}"


def element_info(element: str) -> str:
    """Full info summary for an element."""
    n = _resolve(element)
    if not n:
        return f"unknown element: {element}"
    sym, name, weight, group, period, cat, econfig = _ELEMENTS[n]
    return (f"{n}. {sym} ({name}): weight={weight}, group={group}, "
            f"period={period}, category={cat}, config={econfig}")


ELEMENTS_NL_PATTERNS = [
    (r"(?:atomic\s+)?(?:weight|mass)\s+of\s+(\w+)", 'atomic_weight("{0}")'),
    (r"(?:atomic\s+)?number\s+of\s+(\w+)", 'atomic_number("{0}")'),
    (r"(?:electron\s+)?config(?:uration)?\s+(?:of|for)\s+(\w+)", 'electron_config("{0}")'),
    (r"(?:symbol|chemical symbol)\s+(?:of|for)\s+(\w+)", 'element_symbol("{0}")'),
    (r"(?:what\s+)?element\s+(?:has|is|with)\s+(?:atomic\s+)?number\s+(\d+)", 'element_info("{0}")'),
]

ELEMENTS_FUNCTIONS = {
    "element_symbol": element_symbol,
    "element_name": element_name,
    "atomic_number": atomic_number,
    "atomic_weight": atomic_weight,
    "element_group": element_group,
    "element_period": element_period,
    "element_category": element_category,
    "electron_config": electron_config,
    "element_info": element_info,
}
