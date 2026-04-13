"""
CALM License knowledge backend — open source license facts.

Models confuse MIT vs Apache vs GPL permissions. Stable data.
"""

from __future__ import annotations

_DATA_VERSION = "2025-01"

# (spdx, name, permissions, conditions, limitations, compatible_with)
_LICENSES = {
    "mit": (
        "MIT", "MIT License",
        ["commercial use", "modification", "distribution", "private use"],
        ["include copyright notice"],
        ["no liability", "no warranty"],
        ["apache-2.0", "gpl-2.0", "gpl-3.0", "bsd-2", "bsd-3"],
    ),
    "apache-2.0": (
        "Apache-2.0", "Apache License 2.0",
        ["commercial use", "modification", "distribution", "patent use", "private use"],
        ["include copyright notice", "state changes", "include license"],
        ["no liability", "no warranty", "no trademark use"],
        ["mit", "gpl-3.0", "bsd-2", "bsd-3"],
    ),
    "gpl-2.0": (
        "GPL-2.0", "GNU General Public License v2.0",
        ["commercial use", "modification", "distribution", "private use"],
        ["disclose source", "include license", "same license", "state changes"],
        ["no liability", "no warranty"],
        ["mit", "bsd-2", "bsd-3"],
    ),
    "gpl-3.0": (
        "GPL-3.0", "GNU General Public License v3.0",
        ["commercial use", "modification", "distribution", "patent use", "private use"],
        ["disclose source", "include license", "same license", "state changes"],
        ["no liability", "no warranty"],
        ["mit", "apache-2.0", "bsd-2", "bsd-3", "lgpl-3.0"],
    ),
    "lgpl-3.0": (
        "LGPL-3.0", "GNU Lesser General Public License v3.0",
        ["commercial use", "modification", "distribution", "patent use", "private use"],
        ["disclose source (library)", "include license", "same license (library)"],
        ["no liability", "no warranty"],
        ["mit", "apache-2.0", "bsd-2", "bsd-3", "gpl-3.0"],
    ),
    "bsd-2": (
        "BSD-2-Clause", "BSD 2-Clause \"Simplified\" License",
        ["commercial use", "modification", "distribution", "private use"],
        ["include copyright notice"],
        ["no liability", "no warranty"],
        ["mit", "apache-2.0", "gpl-2.0", "gpl-3.0"],
    ),
    "bsd-3": (
        "BSD-3-Clause", "BSD 3-Clause \"New\" License",
        ["commercial use", "modification", "distribution", "private use"],
        ["include copyright notice", "no endorsement"],
        ["no liability", "no warranty"],
        ["mit", "apache-2.0", "gpl-2.0", "gpl-3.0"],
    ),
    "mpl-2.0": (
        "MPL-2.0", "Mozilla Public License 2.0",
        ["commercial use", "modification", "distribution", "patent use", "private use"],
        ["disclose source (modified files)", "include license"],
        ["no liability", "no warranty", "no trademark use"],
        ["apache-2.0", "gpl-2.0", "gpl-3.0"],
    ),
    "isc": (
        "ISC", "ISC License",
        ["commercial use", "modification", "distribution", "private use"],
        ["include copyright notice"],
        ["no liability", "no warranty"],
        ["mit", "apache-2.0", "gpl-2.0", "gpl-3.0", "bsd-2", "bsd-3"],
    ),
    "unlicense": (
        "Unlicense", "The Unlicense",
        ["commercial use", "modification", "distribution", "private use"],
        [],
        ["no liability", "no warranty"],
        ["mit", "apache-2.0", "bsd-2", "bsd-3"],
    ),
    "cc0": (
        "CC0-1.0", "Creative Commons Zero v1.0 Universal",
        ["commercial use", "modification", "distribution", "private use"],
        [],
        ["no liability", "no warranty", "no patent use"],
        ["mit", "apache-2.0", "bsd-2", "bsd-3"],
    ),
    "agpl-3.0": (
        "AGPL-3.0", "GNU Affero General Public License v3.0",
        ["commercial use", "modification", "distribution", "patent use", "private use"],
        ["disclose source", "include license", "same license", "state changes", "network use is distribution"],
        ["no liability", "no warranty"],
        ["gpl-3.0"],
    ),
}

_ALIASES = {
    "mit": "mit", "bsd": "bsd-2", "bsd2": "bsd-2", "bsd3": "bsd-3",
    "apache": "apache-2.0", "apache2": "apache-2.0",
    "gpl": "gpl-3.0", "gpl2": "gpl-2.0", "gpl3": "gpl-3.0",
    "lgpl": "lgpl-3.0", "lgpl3": "lgpl-3.0",
    "mpl": "mpl-2.0", "mpl2": "mpl-2.0",
    "agpl": "agpl-3.0", "agpl3": "agpl-3.0",
    "cc0": "cc0", "public domain": "unlicense",
}


def _resolve(name: str) -> str | None:
    key = name.strip().lower().replace(" ", "-").replace("_", "-")
    return _ALIASES.get(key, key) if key in _LICENSES or key in _ALIASES else None


def license_info(name: str) -> str:
    """Full summary of a license."""
    key = _resolve(name)
    if not key or key not in _LICENSES:
        return f"unknown license: {name}"
    spdx, full, perms, conds, limits, compat = _LICENSES[key]
    return (f"{spdx} ({full}): permits [{', '.join(perms)}], "
            f"requires [{', '.join(conds) or 'nothing'}], "
            f"limits [{', '.join(limits)}]")


def license_permissions(name: str) -> list:
    """What a license permits."""
    key = _resolve(name)
    if not key or key not in _LICENSES:
        return [f"unknown license: {name}"]
    return _LICENSES[key][2]


def license_conditions(name: str) -> list:
    """What a license requires."""
    key = _resolve(name)
    if not key or key not in _LICENSES:
        return [f"unknown license: {name}"]
    return _LICENSES[key][3]


def license_copyleft(name: str) -> bool:
    """Whether a license is copyleft (requires derivative works use same license)."""
    key = _resolve(name)
    if not key or key not in _LICENSES:
        return False
    return "same license" in " ".join(_LICENSES[key][3])


def license_compatible(license1: str, license2: str) -> str:
    """Check if two licenses are compatible."""
    k1 = _resolve(license1)
    k2 = _resolve(license2)
    if not k1 or k1 not in _LICENSES:
        return f"unknown license: {license1}"
    if not k2 or k2 not in _LICENSES:
        return f"unknown license: {license2}"
    if k1 == k2:
        return f"same license ({k1}): compatible"
    compat1 = _LICENSES[k1][5]
    if k2 in compat1:
        return f"{k1} → {k2}: compatible"
    return f"{k1} → {k2}: may not be compatible (check specific use case)"


LICENSE_FUNCTIONS = {
    "license_info": license_info,
    "license_permissions": license_permissions,
    "license_conditions": license_conditions,
    "license_copyleft": license_copyleft,
    "license_compatible": license_compatible,
}
