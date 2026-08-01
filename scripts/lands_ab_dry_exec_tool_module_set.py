"""Canonical LANDS-AB dry-exec tool-module set (O3 pure pin owner).

Sole authoritative enumeration of dry-exec tool surface paths.
Dependency direction: this module is imported by dry tool + generator;
it must not import either.
"""
from __future__ import annotations

DRY_EXEC_TOOL_ENTRYPOINT = "scripts/lands_ab_packet_dry_exec.py"

# Sorted unique frozen tuple — entrypoint + this owner (self).
DRY_EXEC_TOOL_MODULE_SET: tuple[str, ...] = (
    "scripts/lands_ab_dry_exec_tool_module_set.py",
    "scripts/lands_ab_packet_dry_exec.py",
)
