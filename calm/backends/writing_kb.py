"""Creative writing knowledge backend.

Poetry forms, rhetoric devices, literary terms. Compiled knowledge
for form validation and writing analysis.
"""

from __future__ import annotations

_DATA_VERSION = "2026-04-16"

_POETRY_FORMS = {
    "haiku": {
        "lines": 3, "syllables": [5, 7, 5], "rhyme": None,
        "origin": "Japanese", "description": "3-line nature poem",
    },
    "tanka": {
        "lines": 5, "syllables": [5, 7, 5, 7, 7], "rhyme": None,
        "origin": "Japanese", "description": "5-line extension of haiku",
    },
    "sonnet_shakespearean": {
        "lines": 14, "syllables": None, "rhyme": "ABAB CDCD EFEF GG",
        "origin": "English", "description": "3 quatrains + couplet, iambic pentameter",
    },
    "sonnet_petrarchan": {
        "lines": 14, "syllables": None, "rhyme": "ABBAABBA CDECDE",
        "origin": "Italian", "description": "Octave + sestet",
    },
    "limerick": {
        "lines": 5, "syllables": None, "rhyme": "AABBA",
        "origin": "English", "description": "Humorous, anapestic meter",
    },
    "villanelle": {
        "lines": 19, "syllables": None, "rhyme": "ABA ABA ABA ABA ABA ABAA",
        "origin": "French", "description": "5 tercets + quatrain, 2 refrains",
    },
    "ghazal": {
        "lines": None, "syllables": None, "rhyme": "AA BA CA DA ...",
        "origin": "Arabic/Persian", "description": "Couplets sharing rhyme + refrain",
    },
    "free_verse": {
        "lines": None, "syllables": None, "rhyme": None,
        "origin": "Modern", "description": "No fixed meter, rhyme, or length",
    },
    "blank_verse": {
        "lines": None, "syllables": None, "rhyme": None,
        "origin": "English", "description": "Unrhymed iambic pentameter",
    },
    "couplet": {
        "lines": 2, "syllables": None, "rhyme": "AA",
        "origin": "Universal", "description": "Two rhyming lines",
    },
}

_RHETORIC_DEVICES = {
    "alliteration": {"type": "sound", "description": "Repeated initial consonant sounds",
                     "example": "Peter Piper picked a peck of pickled peppers"},
    "assonance": {"type": "sound", "description": "Repeated vowel sounds within words",
                  "example": "The rain in Spain falls mainly on the plain"},
    "consonance": {"type": "sound", "description": "Repeated consonant sounds (not initial)",
                   "example": "pitter-patter, splitter-splatter"},
    "metaphor": {"type": "meaning", "description": "Direct comparison without like/as",
                 "example": "Time is a thief"},
    "simile": {"type": "meaning", "description": "Comparison using like/as",
               "example": "Brave as a lion"},
    "personification": {"type": "meaning", "description": "Human qualities to non-human",
                        "example": "The wind whispered through the trees"},
    "hyperbole": {"type": "meaning", "description": "Exaggeration for effect",
                  "example": "I've told you a million times"},
    "onomatopoeia": {"type": "sound", "description": "Words that sound like their meaning",
                     "example": "buzz, hiss, splash, murmur"},
    "anaphora": {"type": "structure", "description": "Repeated word/phrase at line starts",
                 "example": "We shall fight... We shall fight... We shall fight..."},
    "chiasmus": {"type": "structure", "description": "Reversed parallel structure",
                 "example": "Ask not what your country can do for you..."},
    "enjambment": {"type": "structure", "description": "Sentence continues past line break",
                   "example": "I think that I shall never see / A poem lovely as a tree"},
    "caesura": {"type": "structure", "description": "Strong pause within a line",
                "example": "To err is human; to forgive, divine"},
}

_NARRATIVE_STRUCTURES = {
    "freytag": {"stages": ["exposition", "rising_action", "climax", "falling_action", "denouement"],
                "description": "Classic dramatic arc (Freytag's pyramid)"},
    "hero_journey": {"stages": ["ordinary_world", "call_to_adventure", "refusal",
                                "mentor", "threshold", "tests", "approach",
                                "ordeal", "reward", "road_back", "resurrection", "return"],
                     "description": "Campbell's monomyth (12 stages)"},
    "three_act": {"stages": ["setup", "confrontation", "resolution"],
                  "description": "Screenplay structure"},
    "kishoten": {"stages": ["ki_intro", "sho_development", "ten_twist", "ketsu_conclusion"],
                 "description": "Japanese 4-act structure (Kishotenketsu)"},
}


def get_poetry_form(name: str) -> dict | None:
    """Look up a poetry form by name."""
    return _POETRY_FORMS.get(name.lower().replace(" ", "_"))


def list_poetry_forms() -> list:
    """List all known poetry forms."""
    return list(_POETRY_FORMS.keys())


def get_rhetoric_device(name: str) -> dict | None:
    """Look up a rhetoric device."""
    return _RHETORIC_DEVICES.get(name.lower())


def list_rhetoric_devices(device_type: str = None) -> list:
    """List rhetoric devices, optionally filtered by type (sound/meaning/structure)."""
    if device_type:
        return [k for k, v in _RHETORIC_DEVICES.items() if v["type"] == device_type]
    return list(_RHETORIC_DEVICES.keys())


def get_narrative_structure(name: str) -> dict | None:
    """Look up a narrative structure."""
    return _NARRATIVE_STRUCTURES.get(name.lower().replace(" ", "_").replace("'", ""))


def list_narrative_structures() -> list:
    """List all known narrative structures."""
    return list(_NARRATIVE_STRUCTURES.keys())


WRITING_KB_FUNCTIONS = {
    "get_poetry_form": get_poetry_form,
    "list_poetry_forms": list_poetry_forms,
    "get_rhetoric_device": get_rhetoric_device,
    "list_rhetoric_devices": list_rhetoric_devices,
    "get_narrative_structure": get_narrative_structure,
    "list_narrative_structures": list_narrative_structures,
}

WRITING_KB_NL_PATTERNS = [
    (r"(?:what is|describe|explain)\s+(?:a )?(?:haiku|sonnet|limerick|villanelle|ghazal|tanka)", "get_poetry_form"),
    (r"(?:list|what are)\s+(?:all )?poetry\s+forms", "list_poetry_forms"),
    (r"(?:what is|explain)\s+(?:a )?(?:alliteration|metaphor|simile|anaphora|chiasmus)", "get_rhetoric_device"),
    (r"(?:list|what are)\s+(?:all )?rhet", "list_rhetoric_devices"),
    (r"(?:what is|explain)\s+(?:the )?(?:hero.s journey|freytag|three act|kishoten)", "get_narrative_structure"),
]
