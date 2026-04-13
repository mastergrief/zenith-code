"""
CALM Chemistry knowledge backend — molecular weights, functional groups, common reactions.

Models hallucinate molecular weights, confuse functional groups.
Extends elements_kb with molecular-level knowledge.
"""

from __future__ import annotations

_DATA_VERSION = "2025-01"

# Common molecule molecular weights (g/mol)
_MOLECULES = {
    "H2O": {"name": "Water", "mw": 18.015, "formula": "H₂O", "state": "liquid"},
    "CO2": {"name": "Carbon Dioxide", "mw": 44.009, "formula": "CO₂", "state": "gas"},
    "NaCl": {"name": "Sodium Chloride", "mw": 58.44, "formula": "NaCl", "state": "solid"},
    "HCl": {"name": "Hydrochloric Acid", "mw": 36.461, "formula": "HCl", "state": "gas/liquid"},
    "NaOH": {"name": "Sodium Hydroxide", "mw": 39.997, "formula": "NaOH", "state": "solid"},
    "H2SO4": {"name": "Sulfuric Acid", "mw": 98.079, "formula": "H₂SO₄", "state": "liquid"},
    "HNO3": {"name": "Nitric Acid", "mw": 63.012, "formula": "HNO₃", "state": "liquid"},
    "NH3": {"name": "Ammonia", "mw": 17.031, "formula": "NH₃", "state": "gas"},
    "CH4": {"name": "Methane", "mw": 16.043, "formula": "CH₄", "state": "gas"},
    "C2H5OH": {"name": "Ethanol", "mw": 46.068, "formula": "C₂H₅OH", "state": "liquid"},
    "CH3OH": {"name": "Methanol", "mw": 32.042, "formula": "CH₃OH", "state": "liquid"},
    "C6H12O6": {"name": "Glucose", "mw": 180.156, "formula": "C₆H₁₂O₆", "state": "solid"},
    "C12H22O11": {"name": "Sucrose", "mw": 342.297, "formula": "C₁₂H₂₂O₁₁", "state": "solid"},
    "CaCO3": {"name": "Calcium Carbonate", "mw": 100.087, "formula": "CaCO₃", "state": "solid"},
    "Fe2O3": {"name": "Iron(III) Oxide", "mw": 159.687, "formula": "Fe₂O₃", "state": "solid"},
    "O2": {"name": "Oxygen", "mw": 31.998, "formula": "O₂", "state": "gas"},
    "N2": {"name": "Nitrogen", "mw": 28.014, "formula": "N₂", "state": "gas"},
    "H2": {"name": "Hydrogen", "mw": 2.016, "formula": "H₂", "state": "gas"},
    "Cl2": {"name": "Chlorine", "mw": 70.906, "formula": "Cl₂", "state": "gas"},
    "KOH": {"name": "Potassium Hydroxide", "mw": 56.105, "formula": "KOH", "state": "solid"},
    "H2O2": {"name": "Hydrogen Peroxide", "mw": 34.014, "formula": "H₂O₂", "state": "liquid"},
    "C3H8": {"name": "Propane", "mw": 44.096, "formula": "C₃H₈", "state": "gas"},
    "C8H18": {"name": "Octane", "mw": 114.228, "formula": "C₈H₁₈", "state": "liquid"},
    "NaHCO3": {"name": "Sodium Bicarbonate", "mw": 84.007, "formula": "NaHCO₃", "state": "solid"},
    "CaO": {"name": "Calcium Oxide", "mw": 56.077, "formula": "CaO", "state": "solid"},
    "MgO": {"name": "Magnesium Oxide", "mw": 40.304, "formula": "MgO", "state": "solid"},
    "SiO2": {"name": "Silicon Dioxide", "mw": 60.083, "formula": "SiO₂", "state": "solid"},
    "Al2O3": {"name": "Aluminum Oxide", "mw": 101.961, "formula": "Al₂O₃", "state": "solid"},
    "C2H4": {"name": "Ethylene", "mw": 28.053, "formula": "C₂H₄", "state": "gas"},
    "C6H6": {"name": "Benzene", "mw": 78.112, "formula": "C₆H₆", "state": "liquid"},
    "CH3COOH": {"name": "Acetic Acid", "mw": 60.052, "formula": "CH₃COOH", "state": "liquid"},
    "ATP": {"name": "Adenosine Triphosphate", "mw": 507.18, "formula": "C₁₀H₁₆N₅O₁₃P₃", "state": "solid"},
    "DNA_A": {"name": "Adenine", "mw": 135.13, "formula": "C₅H₅N₅", "state": "solid"},
    "DNA_T": {"name": "Thymine", "mw": 126.11, "formula": "C₅H₆N₂O₂", "state": "solid"},
    "DNA_G": {"name": "Guanine", "mw": 151.13, "formula": "C₅H₅N₅O", "state": "solid"},
    "DNA_C": {"name": "Cytosine", "mw": 111.10, "formula": "C₄H₅N₃O", "state": "solid"},
}

_FUNCTIONAL_GROUPS = {
    "hydroxyl": {"formula": "-OH", "found_in": "alcohols, phenols", "properties": "polar, H-bonding"},
    "carbonyl": {"formula": "C=O", "found_in": "aldehydes, ketones", "properties": "polar, electrophilic C"},
    "carboxyl": {"formula": "-COOH", "found_in": "carboxylic acids", "properties": "acidic, polar"},
    "amino": {"formula": "-NH2", "found_in": "amines, amino acids", "properties": "basic, polar"},
    "phosphate": {"formula": "-PO4", "found_in": "nucleotides, ATP", "properties": "acidic, energy transfer"},
    "sulfhydryl": {"formula": "-SH", "found_in": "cysteine, thiols", "properties": "forms disulfide bonds"},
    "methyl": {"formula": "-CH3", "found_in": "many organics", "properties": "hydrophobic"},
    "ester": {"formula": "-COO-", "found_in": "fats, flavors", "properties": "hydrolyzable"},
    "ether": {"formula": "-O-", "found_in": "diethyl ether, THF", "properties": "relatively inert"},
    "aldehyde": {"formula": "-CHO", "found_in": "glucose (open chain), formaldehyde", "properties": "reducible"},
    "ketone": {"formula": ">C=O", "found_in": "acetone", "properties": "polar, less reactive than aldehyde"},
    "amide": {"formula": "-CONH2", "found_in": "proteins (peptide bonds)", "properties": "stable, planar"},
    "nitrile": {"formula": "-C≡N", "found_in": "acrylonitrile", "properties": "polar, versatile synthetic"},
    "nitro": {"formula": "-NO2", "found_in": "TNT, nitrobenzene", "properties": "electron-withdrawing"},
    "halide": {"formula": "-X (F, Cl, Br, I)", "found_in": "chloroform, freons", "properties": "polar, reactive"},
}

_PH_INDICATORS = {
    "litmus": {"acid_color": "red", "base_color": "blue", "range": "4.5-8.3"},
    "phenolphthalein": {"acid_color": "colorless", "base_color": "pink", "range": "8.2-10.0"},
    "methyl orange": {"acid_color": "red", "base_color": "yellow", "range": "3.1-4.4"},
    "bromothymol blue": {"acid_color": "yellow", "base_color": "blue", "range": "6.0-7.6"},
    "universal indicator": {"acid_color": "red", "neutral_color": "green", "base_color": "purple", "range": "1-14"},
}


def molecule_info(formula: str) -> dict:
    """Get molecular weight and info for a common molecule."""
    key = str(formula).strip()
    entry = _MOLECULES.get(key)
    if not entry:
        # Try case-insensitive
        for k, v in _MOLECULES.items():
            if k.lower() == key.lower():
                return {"formula": k, **v}
        return {"error": f"Unknown molecule: {formula}", "hint": "Use standard formula like H2O, NaCl, C6H12O6"}
    return {"formula": key, **entry}


def molecular_weight(formula: str) -> float:
    """Molecular weight of a common molecule in g/mol."""
    info = molecule_info(formula)
    return info.get("mw", -1.0)


def functional_group(name: str) -> dict:
    """Get info about an organic functional group."""
    key = str(name).lower().strip()
    entry = _FUNCTIONAL_GROUPS.get(key)
    if not entry:
        return {"error": f"Unknown group: {name}", "valid": list(_FUNCTIONAL_GROUPS.keys())}
    return {"group": key, **entry}


def ph_indicator(name: str) -> dict:
    """Get pH indicator color ranges."""
    key = str(name).lower().strip()
    entry = _PH_INDICATORS.get(key)
    if not entry:
        return {"error": f"Unknown indicator: {name}", "valid": list(_PH_INDICATORS.keys())}
    return {"indicator": key, **entry}


def moles_to_grams(moles: float, formula: str) -> float:
    """Convert moles to grams: mass = moles × molecular_weight."""
    mw = molecular_weight(formula)
    if mw < 0:
        return -1.0
    return round(float(moles) * mw, 4)


def grams_to_moles(grams: float, formula: str) -> float:
    """Convert grams to moles: moles = mass / molecular_weight."""
    mw = molecular_weight(formula)
    if mw <= 0:
        return -1.0
    return round(float(grams) / mw, 6)


def dilution(c1: float, v1: float, c2: float = None, v2: float = None) -> dict:
    """Dilution equation: C1×V1 = C2×V2. Provide 3, get the 4th."""
    vals = [c1, v1, c2, v2]
    nones = sum(1 for v in vals if v is None)
    if nones != 1:
        return {"error": "Provide exactly 3 values"}
    if c2 is None:
        return {"C2": round(float(c1) * float(v1) / float(v2), 4)}
    if v2 is None:
        return {"V2": round(float(c1) * float(v1) / float(c2), 4)}
    if c1 is None:
        return {"C1": round(float(c2) * float(v2) / float(v1), 4)}
    return {"V1": round(float(c2) * float(v2) / float(c1), 4)}


CHEMISTRY_FUNCTIONS = {
    "molecule_info": molecule_info,
    "molecular_weight": molecular_weight,
    "functional_group": functional_group,
    "ph_indicator": ph_indicator,
    "moles_to_grams": moles_to_grams,
    "grams_to_moles": grams_to_moles,
    "dilution": dilution,
}

CHEMISTRY_NL_PATTERNS = [
    (r'(?:molecular weight|molar mass)\s+(?:of|for)\s+(\w+)', 'molecular_weight("{0}")'),
    (r'(?:what is|info about)\s+(\w+)\s+(?:molecule|compound)', 'molecule_info("{0}")'),
    (r'(?:what is|explain)\s+(?:the\s+)?(\w+)\s+(?:functional\s+)?group', 'functional_group("{0}")'),
    (r'convert\s+([\d.]+)\s+moles?\s+(?:of\s+)?(\w+)\s+to\s+grams', 'moles_to_grams({0}, "{1}")'),
    (r'convert\s+([\d.]+)\s+grams?\s+(?:of\s+)?(\w+)\s+to\s+moles', 'grams_to_moles({0}, "{1}")'),
    (r'(?:pH|ph)\s+indicator\s+(\w+)', 'ph_indicator("{0}")'),
]
