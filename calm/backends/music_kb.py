"""
CALM Music theory knowledge backend — note frequencies, intervals, scales, chords.

Models hallucinate frequencies, confuse intervals, give wrong chord formulas.
"""

from __future__ import annotations

import math

_DATA_VERSION = "2025-01"

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_INTERVALS = {
    "unison": {"semitones": 0, "quality": "perfect"},
    "minor second": {"semitones": 1, "quality": "minor", "alias": "half step"},
    "major second": {"semitones": 2, "quality": "major", "alias": "whole step"},
    "minor third": {"semitones": 3, "quality": "minor"},
    "major third": {"semitones": 4, "quality": "major"},
    "perfect fourth": {"semitones": 5, "quality": "perfect"},
    "tritone": {"semitones": 6, "quality": "augmented/diminished", "alias": "augmented fourth"},
    "perfect fifth": {"semitones": 7, "quality": "perfect"},
    "minor sixth": {"semitones": 8, "quality": "minor"},
    "major sixth": {"semitones": 9, "quality": "major"},
    "minor seventh": {"semitones": 10, "quality": "minor"},
    "major seventh": {"semitones": 11, "quality": "major"},
    "octave": {"semitones": 12, "quality": "perfect"},
}

# Chord formulas as semitone intervals from root
_CHORDS = {
    "major": [0, 4, 7],
    "minor": [0, 3, 7],
    "diminished": [0, 3, 6],
    "augmented": [0, 4, 8],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "major7": [0, 4, 7, 11],
    "minor7": [0, 3, 7, 10],
    "dominant7": [0, 4, 7, 10],
    "diminished7": [0, 3, 6, 9],
    "half-diminished7": [0, 3, 6, 10],
    "major9": [0, 4, 7, 11, 14],
    "minor9": [0, 3, 7, 10, 14],
    "add9": [0, 4, 7, 14],
    "power": [0, 7],
}

# Scale formulas as semitone patterns
_SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "natural minor": [0, 2, 3, 5, 7, 8, 10],
    "harmonic minor": [0, 2, 3, 5, 7, 8, 11],
    "melodic minor": [0, 2, 3, 5, 7, 9, 11],
    "pentatonic major": [0, 2, 4, 7, 9],
    "pentatonic minor": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "locrian": [0, 1, 3, 5, 6, 8, 10],
    "chromatic": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "whole tone": [0, 2, 4, 6, 8, 10],
}


def note_frequency(note: str, octave: int = 4) -> float:
    """Frequency in Hz of a note (A4 = 440 Hz, equal temperament)."""
    n = str(note).strip().upper()
    if n.endswith("B") and len(n) > 1 and n[-2].isalpha():
        # Handle flats: Db → C#, etc.
        flat_map = {"DB": "C#", "EB": "D#", "FB": "E", "GB": "F#", "AB": "G#", "BB": "A#"}
        n = flat_map.get(n, n)
    try:
        idx = _NOTE_NAMES.index(n)
    except ValueError:
        return -1.0
    # Semitones from A4
    semitones = (int(octave) - 4) * 12 + (idx - 9)
    return round(440.0 * (2 ** (semitones / 12)), 3)


def interval_info(name: str) -> dict:
    """Get details about a musical interval."""
    key = str(name).lower().strip()
    entry = _INTERVALS.get(key)
    if not entry:
        return {"error": f"Unknown interval: {name}", "valid": list(_INTERVALS.keys())}
    return {"interval": key, **entry}


def chord_notes(root: str, chord_type: str = "major") -> list[str]:
    """Get the notes in a chord (e.g. chord_notes('C', 'major') → ['C', 'E', 'G'])."""
    r = str(root).strip().upper()
    try:
        root_idx = _NOTE_NAMES.index(r)
    except ValueError:
        return [f"Unknown root: {root}"]
    ct = str(chord_type).lower().strip()
    formula = _CHORDS.get(ct)
    if not formula:
        return [f"Unknown chord type: {chord_type}"]
    return [_NOTE_NAMES[(root_idx + s) % 12] for s in formula]


def scale_notes(root: str, scale_type: str = "major") -> list[str]:
    """Get the notes in a scale."""
    r = str(root).strip().upper()
    try:
        root_idx = _NOTE_NAMES.index(r)
    except ValueError:
        return [f"Unknown root: {root}"]
    st = str(scale_type).lower().strip()
    formula = _SCALES.get(st)
    if not formula:
        return [f"Unknown scale type: {scale_type}"]
    return [_NOTE_NAMES[(root_idx + s) % 12] for s in formula]


def semitones_between(note1: str, note2: str) -> int:
    """Count semitones between two notes (ascending)."""
    try:
        idx1 = _NOTE_NAMES.index(str(note1).strip().upper())
        idx2 = _NOTE_NAMES.index(str(note2).strip().upper())
    except ValueError:
        return -1
    return (idx2 - idx1) % 12


def bpm_to_ms(bpm: float) -> float:
    """Convert BPM to milliseconds per beat."""
    b = float(bpm)
    if b <= 0:
        return 0.0
    return round(60000.0 / b, 2)


def transpose(note: str, semitones: int) -> str:
    """Transpose a note by N semitones."""
    try:
        idx = _NOTE_NAMES.index(str(note).strip().upper())
    except ValueError:
        return f"Unknown note: {note}"
    return _NOTE_NAMES[(idx + int(semitones)) % 12]


def list_chord_types() -> list[str]:
    """List all known chord types."""
    return list(_CHORDS.keys())


def list_scale_types() -> list[str]:
    """List all known scale types."""
    return list(_SCALES.keys())


MUSIC_FUNCTIONS = {
    "note_frequency": note_frequency,
    "interval_info": interval_info,
    "chord_notes": chord_notes,
    "scale_notes": scale_notes,
    "semitones_between": semitones_between,
    "bpm_to_ms": bpm_to_ms,
    "transpose": transpose,
    "list_chord_types": list_chord_types,
    "list_scale_types": list_scale_types,
}

MUSIC_NL_PATTERNS = [
    (r'(?:frequency|Hz)\s+(?:of\s+)?([A-Ga-g][#b]?)(\d)', 'note_frequency("{0}", {1})'),
    (r'(?:what notes?|notes? in)\s+(?:a\s+)?([A-Ga-g][#b]?)\s+(major|minor|diminished|augmented|sus[24]|(?:major|minor|dominant|diminished|half-diminished)7?)\s+chord', 'chord_notes("{0}", "{1}")'),
    (r'([A-Ga-g][#b]?)\s+(major|minor|pentatonic|blues|dorian|phrygian|lydian|mixolydian|locrian)\s+scale', 'scale_notes("{0}", "{1}")'),
    (r'semitones?\s+(?:between|from)\s+([A-Ga-g][#b]?)\s+(?:and|to)\s+([A-Ga-g][#b]?)', 'semitones_between("{0}", "{1}")'),
    (r'(\d+)\s*(?:BPM|bpm)\s+(?:to|in)\s+(?:ms|milliseconds)', 'bpm_to_ms({0})'),
    (r'transpose\s+([A-Ga-g][#b]?)\s+(?:up|by)\s+(\d+)\s+semitones?', 'transpose("{0}", {1})'),
]
