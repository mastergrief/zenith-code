"""
CALM package backend — verified package/dependency information.

"What version of X do I have?" — the model guesses. This backend
queries the actual installed packages from pip, npm, and cargo.

Functions: pip_info, pip_list, npm_info, npm_list, cargo_info, cargo_list.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Dict, Optional


def pip_info(package: str) -> dict:
    """Get info about an installed pip package.
    Example: pip_info("requests") → {name, version, location, requires, ...}"""
    package = str(package).strip()
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return {"installed": False, "package": package, "error": result.stderr.strip()[:200]}

        info = {"installed": True}
        for line in result.stdout.splitlines():
            if ":" in line:
                key, _, val = line.partition(":")
                key = key.strip().lower().replace("-", "_")
                info[key] = val.strip()

        # Parse requires into list.
        if "requires" in info and isinstance(info["requires"], str):
            info["requires"] = [r.strip() for r in info["requires"].split(",") if r.strip()]

        return info
    except Exception as e:
        return {"installed": False, "package": package, "error": str(e)[:200]}


def pip_list() -> list:
    """List all installed pip packages. Returns [{name, version}, ...]."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
        return [{"error": result.stderr.strip()[:200]}]
    except Exception as e:
        return [{"error": str(e)[:200]}]


def npm_info(package: str) -> dict:
    """Get info about a locally installed npm package (in current directory).
    Example: npm_info("express") → {name, version, description, ...}"""
    package = str(package).strip()
    try:
        result = subprocess.run(
            ["npm", "ls", package, "--json", "--depth=0"],
            capture_output=True, text=True, timeout=10,
        )
        data = json.loads(result.stdout) if result.stdout else {}
        deps = data.get("dependencies", {})
        if package in deps:
            dep = deps[package]
            return {
                "installed": True,
                "name": package,
                "version": dep.get("version", "unknown"),
                "resolved": dep.get("resolved", ""),
            }
        return {"installed": False, "package": package}
    except FileNotFoundError:
        return {"installed": False, "package": package, "error": "npm not found"}
    except Exception as e:
        return {"installed": False, "package": package, "error": str(e)[:200]}


def npm_list() -> list:
    """List locally installed npm packages. Returns [{name, version}, ...]."""
    try:
        result = subprocess.run(
            ["npm", "ls", "--json", "--depth=0"],
            capture_output=True, text=True, timeout=15,
        )
        data = json.loads(result.stdout) if result.stdout else {}
        deps = data.get("dependencies", {})
        return [{"name": k, "version": v.get("version", "?")} for k, v in deps.items()]
    except FileNotFoundError:
        return [{"error": "npm not found"}]
    except Exception as e:
        return [{"error": str(e)[:200]}]


def cargo_info(crate: str) -> dict:
    """Get info about a cargo crate from local Cargo.toml or lock file.
    Example: cargo_info("serde") → {name, version, features, ...}"""
    crate = str(crate).strip()
    # Try Cargo.lock first (exact versions).
    try:
        with open("Cargo.lock", "r") as f:
            content = f.read()
        import re
        # Parse TOML-like Cargo.lock entries.
        pattern = rf'\[\[package\]\]\s*name\s*=\s*"{re.escape(crate)}"\s*version\s*=\s*"([^"]+)"'
        m = re.search(pattern, content)
        if m:
            return {"installed": True, "name": crate, "version": m.group(1), "source": "Cargo.lock"}
    except FileNotFoundError:
        pass
    except Exception:
        pass

    # Try `cargo metadata`.
    try:
        result = subprocess.run(
            ["cargo", "metadata", "--no-deps", "--format-version=1"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            for pkg in data.get("packages", []):
                if pkg.get("name") == crate:
                    return {
                        "installed": True,
                        "name": pkg["name"],
                        "version": pkg.get("version", "?"),
                        "edition": pkg.get("edition", ""),
                        "features": list(pkg.get("features", {}).keys()),
                        "source": "cargo metadata",
                    }
    except FileNotFoundError:
        pass
    except Exception:
        pass

    return {"installed": False, "crate": crate}


def cargo_list() -> list:
    """List crates from local Cargo.toml dependencies.
    Returns [{name, version_req}, ...]."""
    # Parse Cargo.toml dependencies section.
    try:
        with open("Cargo.toml", "r") as f:
            content = f.read()
        import re
        deps = []
        in_deps = False
        for line in content.splitlines():
            if re.match(r'\[.*dependencies\]', line):
                in_deps = True
                continue
            if line.startswith("[") and in_deps:
                in_deps = False
                continue
            if in_deps and "=" in line:
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and not key.startswith("#"):
                    deps.append({"name": key, "version_req": val[:50]})
        return deps
    except FileNotFoundError:
        return [{"error": "No Cargo.toml in current directory"}]
    except Exception as e:
        return [{"error": str(e)[:200]}]


PACKAGE_FUNCTIONS = {
    "pip_info": pip_info,
    "pip_list": pip_list,
    "npm_info": npm_info,
    "npm_list": npm_list,
    "cargo_info": cargo_info,
    "cargo_list": cargo_list,
}

PACKAGE_NL_PATTERNS = [
    (r'(?:pip|pypi)\s+(?:info|show|details)\s+(?:for|about)\s+(\w[\w-]+)', 'pip_info("{0}")'),
    (r'(?:npm)\s+(?:info|show|details)\s+(?:for|about)\s+(\w[\w-]+)', 'npm_info("{0}")'),
    (r'(?:cargo|crate)\s+(?:info|show|details)\s+(?:for|about)\s+(\w[\w-]+)', 'cargo_info("{0}")'),
]
