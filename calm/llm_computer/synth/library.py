"""Persistent program library — disk-backed entries for autonomous discovery.

A JSONL file at `path` holds one entry per discovered program:
    {"key": "...", "expression": "...", "discovered_at": iso8601,
     "times_invoked": N}

Appended on each new discovery; replayed on load. Loss-free — every
discovery that survives strict validation is a permanent addition.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, Optional


DEFAULT_LIBRARY_PATH = Path("calm/llm_computer/synth/library.jsonl")


@dataclass
class LibraryEntry:
    key: str
    expression: str
    discovered_at: str
    times_invoked: int = 0


class Library:
    """Disk-backed program library with in-memory cache."""

    def __init__(self, path: Path = DEFAULT_LIBRARY_PATH):
        self.path = Path(path)
        self._by_key: Dict[str, LibraryEntry] = {}
        if self.path.exists():
            self._load()

    def _load(self):
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entry = LibraryEntry(**d)
                # Latest entry for each key wins (rewrite-friendly).
                self._by_key[entry.key] = entry

    def lookup(self, key: str) -> Optional[LibraryEntry]:
        entry = self._by_key.get(key)
        if entry is not None:
            entry.times_invoked += 1
            self._append(entry)   # persist invocation count
        return entry

    def register(self, key: str, expression: str) -> LibraryEntry:
        """Register a new (key → expression) mapping. Overwrites if key
        already exists — caller should check existence first if they
        want append-only semantics."""
        entry = LibraryEntry(
            key=key,
            expression=expression,
            discovered_at=datetime.utcnow().isoformat(timespec="seconds"),
            times_invoked=0,
        )
        self._by_key[key] = entry
        self._append(entry)
        return entry

    def _append(self, entry: LibraryEntry):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    def __len__(self) -> int:
        return len(self._by_key)

    def __iter__(self) -> Iterator[LibraryEntry]:
        return iter(self._by_key.values())

    def __contains__(self, key: str) -> bool:
        return key in self._by_key

    def clear(self):
        """Wipe memory state AND delete the on-disk file."""
        self._by_key.clear()
        if self.path.exists():
            self.path.unlink()
