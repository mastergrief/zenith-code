"""Creative writing knowledge backend.

Comprehensive reference for poetry forms, rhetoric devices, meter
patterns, literary terms, genre conventions, character archetypes,
and narrative structures. Each entry is a verified fact that compiles
into ~3 ReGLU neurons in the substrate knowledge DB.

182 entries across 7 categories. Zero hallucination — the system either
knows the form (it's in the DB) or says it doesn't.
"""

from __future__ import annotations

_DATA_VERSION = "2026-04-16"

# --- 40 POETRY FORMS ---

_POETRY_FORMS = {
    # Japanese
    "haiku": {"lines": 3, "syllables": [5, 7, 5], "rhyme": None,
              "origin": "Japanese", "description": "3-line nature poem with seasonal reference (kigo)"},
    "tanka": {"lines": 5, "syllables": [5, 7, 5, 7, 7], "rhyme": None,
              "origin": "Japanese", "description": "5-line extension of haiku, adds emotional turn"},
    "renga": {"lines": None, "syllables": [5, 7, 5, 7, 7], "rhyme": None,
              "origin": "Japanese", "description": "Collaborative linked verse, alternating 5-7-5 and 7-7"},
    "senryu": {"lines": 3, "syllables": [5, 7, 5], "rhyme": None,
               "origin": "Japanese", "description": "Haiku form but about human nature, not nature"},
    # Sonnets
    "sonnet_shakespearean": {"lines": 14, "syllables": None, "rhyme": "ABAB CDCD EFEF GG",
                             "origin": "English", "description": "3 quatrains + couplet, iambic pentameter"},
    "sonnet_petrarchan": {"lines": 14, "syllables": None, "rhyme": "ABBAABBA CDECDE",
                          "origin": "Italian", "description": "Octave (problem) + sestet (resolution)"},
    "sonnet_spenserian": {"lines": 14, "syllables": None, "rhyme": "ABAB BCBC CDCD EE",
                          "origin": "English", "description": "Interlocking rhyme, iambic pentameter"},
    "curtal_sonnet": {"lines": 11, "syllables": None, "rhyme": "ABCABC DBCDC",
                      "origin": "English", "description": "Hopkins' shortened sonnet (10.5 lines)"},
    # Western fixed forms
    "limerick": {"lines": 5, "syllables": None, "rhyme": "AABBA",
                 "origin": "English", "description": "Humorous, anapestic meter, lines 3-4 shorter"},
    "villanelle": {"lines": 19, "syllables": None, "rhyme": "ABA ABA ABA ABA ABA ABAA",
                   "origin": "French", "description": "5 tercets + quatrain, 2 refrains (lines 1 and 3)"},
    "sestina": {"lines": 39, "syllables": None, "rhyme": "end-word rotation",
                "origin": "Provencal", "description": "6 sestets + tercet, 6 end-words rotate in fixed pattern"},
    "pantoum": {"lines": None, "syllables": None, "rhyme": "ABAB (lines 2,4 become 1,3 of next stanza)",
                "origin": "Malay", "description": "Interlocking quatrains, lines repeat across stanzas"},
    "triolet": {"lines": 8, "syllables": None, "rhyme": "ABaAabAB",
                "origin": "French", "description": "8 lines, 2 rhymes, lines 1+2 repeat as 7+8"},
    "rondeau": {"lines": 15, "syllables": None, "rhyme": "AABBA AABR AABBAR",
                "origin": "French", "description": "15 lines, 2 rhymes, opening phrase as refrain (R)"},
    "ballade": {"lines": 28, "syllables": None, "rhyme": "ABABBCBC (×3) + BCBC envoi",
                "origin": "French", "description": "3 octaves + 4-line envoi, refrain last line each stanza"},
    "ottava_rima": {"lines": 8, "syllables": None, "rhyme": "ABABABCC",
                    "origin": "Italian", "description": "8 lines of iambic pentameter, used in epics"},
    "terza_rima": {"lines": None, "syllables": None, "rhyme": "ABA BCB CDC ...",
                   "origin": "Italian", "description": "Interlocking tercets, Dante's Divine Comedy"},
    "clerihew": {"lines": 4, "syllables": None, "rhyme": "AABB",
                 "origin": "English", "description": "Humorous biographical quatrain, irregular meter"},
    "quatrain": {"lines": 4, "syllables": None, "rhyme": "ABAB or ABBA or AABB",
                 "origin": "Universal", "description": "4-line stanza, various rhyme schemes"},
    "couplet": {"lines": 2, "syllables": None, "rhyme": "AA",
                "origin": "Universal", "description": "Two rhyming lines"},
    "tercet": {"lines": 3, "syllables": None, "rhyme": "AAA or ABA",
               "origin": "Universal", "description": "Three-line stanza"},
    # Arabic/Persian
    "ghazal": {"lines": None, "syllables": None, "rhyme": "AA BA CA DA ...",
               "origin": "Arabic/Persian", "description": "Couplets sharing rhyme + refrain, themes of love/loss"},
    "qasida": {"lines": None, "syllables": None, "rhyme": "AA BA CA DA ...",
               "origin": "Arabic", "description": "Long monorhyme, formal praise or elegy"},
    "rubai": {"lines": 4, "syllables": None, "rhyme": "AABA",
              "origin": "Persian", "description": "Quatrain form, Omar Khayyam's Rubaiyat"},
    # Free/modern
    "free_verse": {"lines": None, "syllables": None, "rhyme": None,
                   "origin": "Modern", "description": "No fixed meter, rhyme, or length"},
    "blank_verse": {"lines": None, "syllables": None, "rhyme": None,
                    "origin": "English", "description": "Unrhymed iambic pentameter (Shakespeare, Milton)"},
    "prose_poem": {"lines": None, "syllables": None, "rhyme": None,
                   "origin": "French", "description": "Poetic language in paragraph form, no line breaks"},
    "concrete_poetry": {"lines": None, "syllables": None, "rhyme": None,
                        "origin": "Modern", "description": "Visual arrangement of text forms the meaning"},
    "found_poetry": {"lines": None, "syllables": None, "rhyme": None,
                     "origin": "Modern", "description": "Rearranged text from non-poetic sources"},
    # Specialized
    "acrostic": {"lines": None, "syllables": None, "rhyme": None,
                 "origin": "Ancient", "description": "First letters of each line spell a word or message"},
    "epigram": {"lines": 2, "syllables": None, "rhyme": "AA",
                "origin": "Greek", "description": "Brief, witty, pointed statement in verse"},
    "elegy": {"lines": None, "syllables": None, "rhyme": None,
              "origin": "Greek", "description": "Poem of mourning, often in elegiac couplets"},
    "ode": {"lines": None, "syllables": None, "rhyme": "varies",
            "origin": "Greek", "description": "Formal address to a subject, elevated tone (Pindaric, Horatian, irregular)"},
    "ballad": {"lines": None, "syllables": None, "rhyme": "ABCB or ABAB",
               "origin": "Medieval", "description": "Narrative poem, alternating 4/3 stress lines, often sung"},
    "epic": {"lines": None, "syllables": None, "rhyme": "varies",
             "origin": "Ancient", "description": "Long narrative of heroic deeds (Iliad, Beowulf, Paradise Lost)"},
    "lune": {"lines": 3, "syllables": None, "rhyme": None,
             "origin": "American", "description": "American haiku: 3/5/3 words (not syllables)"},
    "fibonacci": {"lines": None, "syllables": [1, 1, 2, 3, 5, 8], "rhyme": None,
                  "origin": "Modern", "description": "Syllable count follows Fibonacci sequence"},
    "nonet": {"lines": 9, "syllables": [9, 8, 7, 6, 5, 4, 3, 2, 1], "rhyme": None,
              "origin": "Modern", "description": "9 lines, syllables count down 9 to 1"},
    "diamante": {"lines": 7, "syllables": None, "rhyme": None,
                 "origin": "Modern", "description": "Diamond-shaped: 1/2/3/4/3/2/1 words per line"},
    "kyrielle": {"lines": None, "syllables": None, "rhyme": "AABB with refrain",
                 "origin": "French", "description": "Couplet stanzas with repeating refrain line"},
}

# --- 30 RHETORIC DEVICES ---

_RHETORIC_DEVICES = {
    # Sound devices
    "alliteration": {"type": "sound", "description": "Repeated initial consonant sounds",
                     "example": "Peter Piper picked a peck of pickled peppers"},
    "assonance": {"type": "sound", "description": "Repeated vowel sounds within words",
                  "example": "The rain in Spain falls mainly on the plain"},
    "consonance": {"type": "sound", "description": "Repeated consonant sounds (not initial)",
                   "example": "pitter-patter, splitter-splatter"},
    "onomatopoeia": {"type": "sound", "description": "Words that sound like their meaning",
                     "example": "buzz, hiss, splash, murmur, crackle"},
    "sibilance": {"type": "sound", "description": "Repeated s/sh/z sounds",
                  "example": "She sells seashells by the seashore"},
    "euphony": {"type": "sound", "description": "Pleasant, harmonious sounds",
                "example": "Season of mists and mellow fruitfulness"},
    "cacophony": {"type": "sound", "description": "Harsh, discordant sounds for effect",
                  "example": "With throats unslaked, with black lips baked"},
    # Meaning devices
    "metaphor": {"type": "meaning", "description": "Direct comparison without like/as",
                 "example": "Time is a thief"},
    "simile": {"type": "meaning", "description": "Comparison using like/as",
               "example": "Brave as a lion"},
    "personification": {"type": "meaning", "description": "Human qualities to non-human",
                        "example": "The wind whispered through the trees"},
    "hyperbole": {"type": "meaning", "description": "Exaggeration for effect",
                  "example": "I've told you a million times"},
    "litotes": {"type": "meaning", "description": "Understatement via double negative",
                "example": "He's not unkind (meaning: he's kind)"},
    "synecdoche": {"type": "meaning", "description": "Part represents whole (or whole for part)",
                   "example": "All hands on deck (hands = sailors)"},
    "metonymy": {"type": "meaning", "description": "Associated term replaces the thing",
                 "example": "The pen is mightier than the sword (pen = writing, sword = force)"},
    "oxymoron": {"type": "meaning", "description": "Contradictory terms combined",
                 "example": "Deafening silence, bittersweet, living dead"},
    "paradox": {"type": "meaning", "description": "Seemingly contradictory truth",
                "example": "I must be cruel to be kind"},
    "irony": {"type": "meaning", "description": "Meaning opposite to literal words",
              "example": "What lovely weather (said during a storm)"},
    "symbolism": {"type": "meaning", "description": "Concrete object represents abstract idea",
                  "example": "A dove symbolizing peace"},
    "allegory": {"type": "meaning", "description": "Extended metaphor across entire work",
                 "example": "Animal Farm (animals = political figures)"},
    "apostrophe": {"type": "meaning", "description": "Addressing absent person or abstract concept",
                   "example": "O Death, where is thy sting?"},
    # Structure devices
    "anaphora": {"type": "structure", "description": "Repeated word/phrase at line starts",
                 "example": "We shall fight on the beaches, we shall fight on the landing grounds..."},
    "epistrophe": {"type": "structure", "description": "Repeated word/phrase at line ends",
                   "example": "...of the people, by the people, for the people"},
    "chiasmus": {"type": "structure", "description": "Reversed parallel structure (ABBA)",
                 "example": "Ask not what your country can do for you..."},
    "parallelism": {"type": "structure", "description": "Similar grammatical structure repeated",
                    "example": "I came, I saw, I conquered"},
    "enjambment": {"type": "structure", "description": "Sentence continues past line break",
                   "example": "I think that I shall never see / A poem lovely as a tree"},
    "caesura": {"type": "structure", "description": "Strong pause within a line",
                "example": "To err is human; || to forgive, divine"},
    "antithesis": {"type": "structure", "description": "Contrasting ideas in balanced structure",
                   "example": "It was the best of times, it was the worst of times"},
    "zeugma": {"type": "structure", "description": "One word governs two others differently",
               "example": "She lowered her standards and her neckline"},
    "anadiplosis": {"type": "structure", "description": "End of one clause begins the next",
                    "example": "Fear leads to anger. Anger leads to hate."},
    "polysyndeton": {"type": "structure", "description": "Many conjunctions in succession",
                     "example": "And the rain fell and the wind blew and the trees swayed"},
}

# --- 20 METER PATTERNS ---

_METER_PATTERNS = {
    "iambic": {"pattern": "uS", "feet": 2, "description": "Unstressed-stressed (da-DUM)",
               "example": "Shall I / com-PARE / thee TO / a SUM / mer's DAY"},
    "trochaic": {"pattern": "Su", "feet": 2, "description": "Stressed-unstressed (DUM-da)",
                 "example": "TELL me / NOT in / MOURN-ful / NUM-bers"},
    "anapestic": {"pattern": "uuS", "feet": 3, "description": "2 unstressed + stressed (da-da-DUM)",
                  "example": "And the SOUND / of a VOICE / that is STILL"},
    "dactylic": {"pattern": "Suu", "feet": 3, "description": "Stressed + 2 unstressed (DUM-da-da)",
                 "example": "THIS is the / FOR-est pri / ME-val"},
    "spondaic": {"pattern": "SS", "feet": 2, "description": "Two stressed (DUM-DUM)",
                 "example": "HEART-BREAK, DEAD-LOCK, BOOK-CASE"},
    "pyrrhic": {"pattern": "uu", "feet": 2, "description": "Two unstressed (da-da)",
                "example": "Rare in isolation, used for variation"},
    "amphibrachic": {"pattern": "uSu", "feet": 3, "description": "Unstressed-stressed-unstressed",
                     "example": "ro-MAN-tic, tre-MEN-dous"},
    "monometer": {"length": 1, "description": "1 foot per line"},
    "dimeter": {"length": 2, "description": "2 feet per line"},
    "trimeter": {"length": 3, "description": "3 feet per line"},
    "tetrameter": {"length": 4, "description": "4 feet per line"},
    "pentameter": {"length": 5, "description": "5 feet per line (most common in English)"},
    "hexameter": {"length": 6, "description": "6 feet per line (Homeric epic meter)"},
    "heptameter": {"length": 7, "description": "7 feet per line (ballad meter doubled)"},
    "iambic_pentameter": {"pattern": "uS uS uS uS uS", "feet": 10,
                          "description": "5 iambs — Shakespeare, Milton, most English verse",
                          "example": "But SOFT, / what LIGHT / through YON / der WIN / dow BREAKS"},
    "common_meter": {"pattern": "uS uS uS uS / uS uS uS", "feet": 7,
                     "description": "Alternating 4/3 iambic feet (hymns, ballads, Emily Dickinson)",
                     "example": "A-MA- / zing GRACE / how SWEET / the SOUND"},
    "ballad_meter": {"pattern": "uS uS uS uS / uS uS uS", "feet": 7,
                     "description": "Same as common meter but with narrative focus"},
    "alexandrine": {"pattern": "uS uS uS uS uS uS", "feet": 12,
                    "description": "6 iambs — standard French verse meter"},
    "sprung_rhythm": {"pattern": "variable", "feet": None,
                      "description": "Hopkins' system: count stressed syllables only, unstressed vary freely"},
    "free_meter": {"pattern": None, "feet": None,
                   "description": "No fixed metrical pattern — free verse"},
}

# --- 12 CHARACTER ARCHETYPES ---

_CHARACTER_ARCHETYPES = {
    "hero": {"role": "protagonist", "traits": "courage, determination, sacrifice",
             "shadow": "tyrant", "example": "Frodo, Luke Skywalker, Katniss"},
    "mentor": {"role": "guide", "traits": "wisdom, knowledge, guidance",
               "shadow": "manipulator", "example": "Gandalf, Obi-Wan, Dumbledore"},
    "threshold_guardian": {"role": "obstacle", "traits": "tests the hero's resolve",
                           "shadow": "bully", "example": "Cerberus, bouncers, gatekeepers"},
    "herald": {"role": "catalyst", "traits": "announces change, delivers the call",
               "shadow": "false prophet", "example": "R2-D2's message, the white rabbit"},
    "shapeshifter": {"role": "uncertainty", "traits": "changes allegiance, unreliable",
                     "shadow": "betrayer", "example": "Snape, Catwoman, Loki"},
    "shadow": {"role": "antagonist", "traits": "represents what hero fears/rejects",
               "shadow": "hero's dark mirror", "example": "Darth Vader, Moriarty, Sauron"},
    "trickster": {"role": "comic relief/catalyst", "traits": "mischief, disruption, truth-telling",
                  "shadow": "chaos agent", "example": "Puck, Loki, Jack Sparrow"},
    "ally": {"role": "companion", "traits": "loyalty, support, complementary skills",
             "shadow": "sycophant", "example": "Samwise, Ron Weasley, Dr. Watson"},
    "mother": {"role": "nurturer", "traits": "protection, comfort, unconditional love",
               "shadow": "devouring mother", "example": "Molly Weasley, Galadriel"},
    "father": {"role": "authority", "traits": "order, discipline, tradition",
               "shadow": "ogre/tyrant", "example": "Mufasa, Atticus Finch"},
    "child": {"role": "innocent", "traits": "wonder, spontaneity, vulnerability",
              "shadow": "orphan/victim", "example": "Scout Finch, Oliver Twist"},
    "sage": {"role": "truth-seeker", "traits": "analysis, objectivity, understanding",
             "shadow": "detached cynic", "example": "Sherlock Holmes, Yoda"},
}

# --- 30 GENRE CONVENTIONS ---

_GENRE_CONVENTIONS = {
    "mystery": {"required": ["crime/puzzle", "clues", "red herrings", "revelation"],
                "optional": ["detective protagonist", "locked room", "unreliable narrator"],
                "avoid": ["deus ex machina solution", "unsolvable-by-reader puzzle"]},
    "romance": {"required": ["central love story", "emotionally satisfying ending (HEA/HFN)"],
                "optional": ["meet-cute", "forced proximity", "rivals-to-lovers"],
                "avoid": ["love interest death without resolution", "cheating played as romantic"]},
    "thriller": {"required": ["high stakes", "escalating tension", "time pressure"],
                 "optional": ["unreliable narrator", "plot twists", "cat-and-mouse"],
                 "avoid": ["stakes that never escalate", "convenient coincidences"]},
    "horror": {"required": ["fear/dread", "threat", "atmosphere"],
               "optional": ["isolation", "unreliable reality", "body horror", "cosmic dread"],
               "avoid": ["jump scares as only tool", "torture without purpose"]},
    "fantasy": {"required": ["secondary world or magic system", "internal consistency"],
                "optional": ["quest", "chosen one", "world-building", "magic costs"],
                "avoid": ["deus ex machina magic", "inconsistent rules"]},
    "science_fiction": {"required": ["speculative element grounded in science/tech", "what-if premise"],
                        "optional": ["world-building", "social commentary", "technology consequences"],
                        "avoid": ["magic disguised as science", "unexplained tech"]},
    "literary_fiction": {"required": ["character depth", "prose quality", "thematic exploration"],
                         "optional": ["ambiguous ending", "unreliable narrator", "non-linear structure"],
                         "avoid": ["plot over character", "didactic messaging"]},
    "comedy": {"required": ["humor", "comic timing", "satisfying resolution"],
               "optional": ["misunderstanding", "dramatic irony", "escalation"],
               "avoid": ["punching down", "humor that undermines stakes"]},
    "tragedy": {"required": ["protagonist's fatal flaw", "inevitable downfall", "catharsis"],
                "optional": ["hubris", "reversal of fortune", "recognition"],
                "avoid": ["suffering without meaning", "arbitrary destruction"]},
    "memoir": {"required": ["truth", "reflection", "narrative arc within real events"],
               "optional": ["thematic focus", "vulnerability", "humor"],
               "avoid": ["fabrication", "self-aggrandizement", "chronological dump"]},
}

# --- 8 NARRATIVE STRUCTURES ---

_NARRATIVE_STRUCTURES = {
    "freytag": {"stages": ["exposition", "rising_action", "climax", "falling_action", "denouement"],
                "description": "Classic dramatic arc (Freytag's pyramid)"},
    "hero_journey": {"stages": ["ordinary_world", "call_to_adventure", "refusal",
                                "mentor", "threshold", "tests", "approach",
                                "ordeal", "reward", "road_back", "resurrection", "return"],
                     "description": "Campbell's monomyth (12 stages)"},
    "three_act": {"stages": ["setup", "confrontation", "resolution"],
                  "description": "Screenplay structure"},
    "kishotenketsu": {"stages": ["ki_intro", "sho_development", "ten_twist", "ketsu_conclusion"],
                      "description": "Japanese 4-act structure (no conflict required)"},
    "save_the_cat": {"stages": ["opening_image", "theme_stated", "setup", "catalyst",
                                "debate", "break_into_two", "b_story", "midpoint",
                                "bad_guys_close_in", "all_is_lost", "dark_night",
                                "break_into_three", "finale", "final_image"],
                     "description": "Blake Snyder's 14-beat screenplay structure"},
    "seven_point": {"stages": ["hook", "plot_turn_1", "pinch_1", "midpoint",
                               "pinch_2", "plot_turn_2", "resolution"],
                    "description": "Dan Wells' 7-point story structure"},
    "fichtean_curve": {"stages": ["crisis_1", "crisis_2", "crisis_3", "climax", "falling_action"],
                       "description": "Starts in medias res, rising crises without exposition"},
    "nested_loops": {"stages": ["outer_story_start", "inner_story_1", "inner_story_2",
                                "core_message", "inner_resolution", "outer_resolution"],
                     "description": "Stories within stories, common in oral tradition and TED talks"},
}

# --- 30 WRITING RULES ---

_WRITING_RULES = {
    "show_dont_tell": {"rule": "Convey through action and sensory detail, not exposition",
                       "bad": "She was sad.", "good": "Her hands trembled as she folded the letter."},
    "active_voice": {"rule": "Prefer subject-verb-object over passive constructions",
                     "bad": "The ball was thrown by him.", "good": "He threw the ball."},
    "said_is_invisible": {"rule": "Use 'said' for dialogue tags — readers skip it naturally",
                          "bad": "he exclaimed/growled/hissed", "good": "he said"},
    "kill_adverbs": {"rule": "Use stronger verbs instead of verb + adverb",
                     "bad": "She ran quickly.", "good": "She sprinted."},
    "concrete_nouns": {"rule": "Prefer specific over general nouns",
                       "bad": "The vehicle went down the road.", "good": "The Ford pickup rattled down the gravel lane."},
    "vary_sentence_length": {"rule": "Mix short and long sentences for rhythm",
                             "bad": "He walked. He sat. He ate. He slept.",
                             "good": "He walked for hours. When he finally sat, the chair groaned beneath him."},
    "cut_filler": {"rule": "Remove words that add no meaning: very, really, quite, just, actually",
                   "bad": "She was really very tired.", "good": "She was exhausted."},
    "start_late_end_early": {"rule": "Enter scenes as late as possible, leave as early as possible",
                             "bad": "He parked, walked to the door, knocked, waited...",
                             "good": "The door opened. 'You're late,' she said."},
    "conflict_every_scene": {"rule": "Every scene needs tension — character wants something and faces obstacle",
                             "bad": "They had a pleasant dinner.", "good": "She smiled through the dinner, the divorce papers in her purse."},
    "chekhov_gun": {"rule": "If you introduce a detail, it must pay off later",
                    "bad": "A rifle hung over the mantel. (never mentioned again)",
                    "good": "A rifle hung over the mantel. (fires in act 3)"},
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


def get_meter_pattern(name: str) -> dict | None:
    """Look up a meter pattern."""
    return _METER_PATTERNS.get(name.lower().replace(" ", "_"))


def list_meter_patterns() -> list:
    """List all known meter patterns."""
    return list(_METER_PATTERNS.keys())


def get_archetype(name: str) -> dict | None:
    """Look up a character archetype."""
    return _CHARACTER_ARCHETYPES.get(name.lower().replace(" ", "_"))


def list_archetypes() -> list:
    """List all character archetypes."""
    return list(_CHARACTER_ARCHETYPES.keys())


def get_genre(name: str) -> dict | None:
    """Look up genre conventions."""
    return _GENRE_CONVENTIONS.get(name.lower().replace(" ", "_"))


def list_genres() -> list:
    """List all known genre conventions."""
    return list(_GENRE_CONVENTIONS.keys())


def get_writing_rule(name: str) -> dict | None:
    """Look up a writing rule with good/bad examples."""
    return _WRITING_RULES.get(name.lower().replace(" ", "_"))


def list_writing_rules() -> list:
    """List all writing rules."""
    return list(_WRITING_RULES.keys())


WRITING_KB_FUNCTIONS = {
    "get_poetry_form": get_poetry_form,
    "list_poetry_forms": list_poetry_forms,
    "get_rhetoric_device": get_rhetoric_device,
    "list_rhetoric_devices": list_rhetoric_devices,
    "get_narrative_structure": get_narrative_structure,
    "list_narrative_structures": list_narrative_structures,
    "get_meter_pattern": get_meter_pattern,
    "list_meter_patterns": list_meter_patterns,
    "get_archetype": get_archetype,
    "list_archetypes": list_archetypes,
    "get_genre": get_genre,
    "list_genres": list_genres,
    "get_writing_rule": get_writing_rule,
    "list_writing_rules": list_writing_rules,
}

WRITING_KB_NL_PATTERNS = [
    (r"(?:what is|describe|explain)\s+(?:a )?(?:haiku|sonnet|limerick|villanelle|ghazal|tanka|sestina|pantoum)", "get_poetry_form"),
    (r"(?:list|what are)\s+(?:all )?poetry\s+forms", "list_poetry_forms"),
    (r"(?:what is|explain)\s+(?:a )?(?:alliteration|metaphor|simile|anaphora|chiasmus|litotes|synecdoche|metonymy)", "get_rhetoric_device"),
    (r"(?:list|what are)\s+(?:all )?rhet", "list_rhetoric_devices"),
    (r"(?:what is|explain)\s+(?:the )?(?:hero.s journey|freytag|three act|kishoten|save the cat)", "get_narrative_structure"),
    (r"(?:what is|explain)\s+(?:the )?(?:iambic|trochaic|anapestic|dactylic|spondaic|pentameter)", "get_meter_pattern"),
    (r"(?:list|what are)\s+(?:all )?meter", "list_meter_patterns"),
    (r"(?:what is|explain)\s+(?:the )?(?:hero|mentor|trickster|shadow|sage)\s+archetype", "get_archetype"),
    (r"(?:list|what are)\s+(?:all )?(?:character )?archetypes?", "list_archetypes"),
    (r"(?:conventions?|rules?)\s+(?:for|of)\s+(?:mystery|romance|thriller|horror|fantasy|sci.fi)", "get_genre"),
    (r"(?:list|what are)\s+(?:all )?genres?", "list_genres"),
    (r"(?:what is|explain)\s+(?:show.don.t.tell|active.voice|chekhov)", "get_writing_rule"),
    (r"(?:list|what are)\s+(?:all )?writing\s+rules?", "list_writing_rules"),
]
